using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace ImageQualityEnhancer.Services;

public sealed record WorkerOptions
{
    /// <summary>Python interpreter. Overridable with the IQE_PYTHON environment variable.</summary>
    public string PythonExecutable { get; init; } =
        System.Environment.GetEnvironmentVariable("IQE_PYTHON")
        ?? (OperatingSystem.IsWindows() ? "python" : "python3");

    public string ScriptPath { get; init; } = Path.Combine(AppContext.BaseDirectory, "worker", "main.py");

    /// <summary>Extra environment variables for the worker process.</summary>
    public IReadOnlyDictionary<string, string> Environment { get; init; } = new Dictionary<string, string>();

    public TimeSpan StartTimeout { get; init; } = TimeSpan.FromSeconds(30);
    public TimeSpan ShutdownTimeout { get; init; } = TimeSpan.FromSeconds(5);
}

/// <summary>The worker answered a request with ok=false (e.g. unsupported image, corrupt PDF).</summary>
public sealed class WorkerException(string code, string message) : Exception(message)
{
    public string Code { get; } = code;
}

/// <summary>The worker process is not running (failed to start, crashed, or was stopped).</summary>
public sealed class WorkerUnavailableException(string message, Exception? inner = null) : Exception(message, inner);

/// <summary>
/// Owns one local Python worker process and exchanges JSON-lines messages with it (see worker/protocol.py).
/// Thread-safe: requests may be sent from any thread; responses are matched by id.
/// </summary>
public sealed class PythonWorkerClient : IAsyncDisposable
{
    private const int StderrTailLines = 50;

    // Same naming/enums as persisted state, but one message per line.
    private static readonly JsonSerializerOptions RequestJsonOptions =
        new(JobStateStore.JsonOptions) { WriteIndented = false };

    private readonly WorkerOptions _options;
    private readonly ConcurrentDictionary<string, PendingRequest> _pending = new();
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly Queue<string> _stderrTail = new();
    private readonly TaskCompletionSource _ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Process? _process;
    private long _nextId;

    public PythonWorkerClient(WorkerOptions? options = null)
    {
        _options = options ?? new WorkerOptions();
    }

    public bool IsRunning => _process is { HasExited: false };

    /// <summary>
    /// PID of the Python interpreter doing the work, as reported by the worker. This can differ from the launched
    /// process when "python" is a launcher (e.g. the Windows Python install manager) that starts a child interpreter.
    /// </summary>
    public int? WorkerPid { get; private set; }

    /// <summary>Last lines the worker wrote to stderr, for diagnostics when it fails.</summary>
    public string StderrTail
    {
        get
        {
            lock (_stderrTail)
            {
                return string.Join(Environment.NewLine, _stderrTail);
            }
        }
    }

    public async Task StartAsync(CancellationToken ct = default)
    {
        if (_process is not null)
        {
            throw new InvalidOperationException("Worker already started.");
        }

        if (!File.Exists(_options.ScriptPath))
        {
            throw new WorkerUnavailableException($"Worker script not found: {_options.ScriptPath}");
        }

        var utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        var psi = new ProcessStartInfo
        {
            FileName = _options.PythonExecutable,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardInputEncoding = utf8,
            StandardOutputEncoding = utf8,
            StandardErrorEncoding = utf8,
            WorkingDirectory = Path.GetDirectoryName(_options.ScriptPath)!,
        };
        psi.ArgumentList.Add("-B"); // don't write .pyc files next to the shipped worker
        psi.ArgumentList.Add(_options.ScriptPath);
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        foreach (var (key, value) in _options.Environment)
        {
            psi.Environment[key] = value;
        }

        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        try
        {
            process.Start();
        }
        catch (Exception ex)
        {
            process.Dispose();
            throw new WorkerUnavailableException(
                $"Could not start Python ('{_options.PythonExecutable}'). Is Python installed?", ex);
        }

        _process = process;
        _ = Task.Run(() => ReadStdoutAsync(process));
        _ = Task.Run(() => ReadStderrAsync(process));

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(_options.StartTimeout);
        try
        {
            await _ready.Task.WaitAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            Kill();
            throw new WorkerUnavailableException("Worker did not become ready in time.");
        }
    }

    /// <summary>Sends a command and waits for its result. Progress messages for this request go to <paramref name="progress"/>.</summary>
    public async Task<JsonElement> SendAsync(
        string command,
        object? args = null,
        IProgress<JsonElement>? progress = null,
        CancellationToken ct = default)
    {
        var process = _process;
        if (process is null || process.HasExited)
        {
            throw new WorkerUnavailableException("Worker is not running.");
        }

        var id = Interlocked.Increment(ref _nextId).ToString();
        var pending = new PendingRequest(progress);
        _pending[id] = pending;

        try
        {
            var line = JsonSerializer.Serialize(new { id, cmd = command, args }, RequestJsonOptions);
            await _writeLock.WaitAsync(ct);
            try
            {
                await process.StandardInput.WriteLineAsync(line.AsMemory(), ct);
                await process.StandardInput.FlushAsync(ct);
            }
            catch (IOException ex)
            {
                throw new WorkerUnavailableException("Worker stopped while sending a request.", ex);
            }
            finally
            {
                _writeLock.Release();
            }

            // Cancelling only stops waiting; interrupting the work itself is done by stopping the worker.
            return await pending.Completion.Task.WaitAsync(ct);
        }
        finally
        {
            _pending.TryRemove(id, out _);
        }
    }

    /// <summary>Asks the worker to exit; kills it if it does not exit within the shutdown timeout.</summary>
    public async Task StopAsync()
    {
        var process = _process;
        if (process is null || process.HasExited)
        {
            return;
        }

        try
        {
            using var cts = new CancellationTokenSource(_options.ShutdownTimeout);
            await SendAsync("shutdown", ct: cts.Token);
            await process.WaitForExitAsync(cts.Token);
        }
        catch (Exception ex) when (ex is OperationCanceledException or WorkerException or WorkerUnavailableException)
        {
            Kill();
        }
    }

    /// <summary>Terminates the worker immediately (used for cancel when the current image cannot be interrupted).</summary>
    public void Kill()
    {
        try
        {
            if (_process is { HasExited: false })
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
            // Already exited.
        }
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _process?.Dispose();
        _writeLock.Dispose();
    }

    private async Task ReadStdoutAsync(Process process)
    {
        try
        {
            while (await process.StandardOutput.ReadLineAsync() is { } line)
            {
                HandleMessage(line);
            }
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException or InvalidOperationException)
        {
            // Pipe closed; handled below.
        }

        await process.WaitForExitAsync();
        var reason = new WorkerUnavailableException(
            $"Worker process exited (code {SafeExitCode(process)}).{FormatStderr()}");
        _ready.TrySetException(reason);
        foreach (var (_, pending) in _pending)
        {
            pending.Completion.TrySetException(reason);
        }
    }

    private async Task ReadStderrAsync(Process process)
    {
        try
        {
            while (await process.StandardError.ReadLineAsync() is { } line)
            {
                lock (_stderrTail)
                {
                    _stderrTail.Enqueue(line);
                    while (_stderrTail.Count > StderrTailLines)
                    {
                        _stderrTail.Dequeue();
                    }
                }
            }
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException or InvalidOperationException)
        {
        }
    }

    private void HandleMessage(string line)
    {
        JsonElement message;
        try
        {
            message = JsonDocument.Parse(line).RootElement.Clone();
        }
        catch (JsonException)
        {
            // Stray non-protocol output (e.g. a library printing to stdout). Keep it for diagnostics only.
            lock (_stderrTail)
            {
                _stderrTail.Enqueue("[stdout] " + line);
            }

            return;
        }

        var type = message.TryGetProperty("type", out var t) ? t.GetString() : null;
        if (type == "ready")
        {
            if (message.TryGetProperty("pid", out var pid) && pid.TryGetInt32(out var workerPid))
            {
                WorkerPid = workerPid;
            }

            _ready.TrySetResult();
            return;
        }

        var id = message.TryGetProperty("id", out var idElement) && idElement.ValueKind == JsonValueKind.String
            ? idElement.GetString()
            : null;
        if (id is null || !_pending.TryGetValue(id, out var pending))
        {
            return;
        }

        if (type == "progress")
        {
            if (message.TryGetProperty("data", out var data))
            {
                pending.Progress?.Report(data);
            }

            return;
        }

        if (message.TryGetProperty("ok", out var ok) && ok.ValueKind == JsonValueKind.True)
        {
            pending.Completion.TrySetResult(message.TryGetProperty("result", out var result) ? result : default);
        }
        else
        {
            var error = message.GetProperty("error");
            pending.Completion.TrySetException(new WorkerException(
                error.GetProperty("code").GetString() ?? "unknown",
                error.GetProperty("message").GetString() ?? "Unknown worker error."));
        }
    }

    private string FormatStderr()
    {
        var tail = StderrTail;
        return tail.Length == 0 ? "" : Environment.NewLine + tail;
    }

    private static string SafeExitCode(Process process)
    {
        try
        {
            return process.ExitCode.ToString();
        }
        catch (InvalidOperationException)
        {
            return "unknown";
        }
    }

    private sealed class PendingRequest(IProgress<JsonElement>? progress)
    {
        public IProgress<JsonElement>? Progress { get; } = progress;

        public TaskCompletionSource<JsonElement> Completion { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
