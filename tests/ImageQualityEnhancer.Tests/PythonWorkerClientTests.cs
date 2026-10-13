using System.Diagnostics;
using System.Text.Json;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>
/// Integration tests against the real worker script, which the app project copies to worker/ in the output folder.
/// Requires Python on PATH (or IQE_PYTHON).
/// </summary>
public class PythonWorkerClientTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(30);

    [Fact]
    public async Task Ping_RoundTripsUnicode()
    {
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var result = await client.SendAsync("ping", new { echo = "Réaction – 化学" }).WaitAsync(Timeout);

        Assert.True(result.GetProperty("pong").GetBoolean());
        Assert.Equal("Réaction – 化学", result.GetProperty("echo").GetString());
        Assert.Equal(client.WorkerPid, result.GetProperty("pid").GetInt32());
    }

    [Fact]
    public async Task ConcurrentRequests_AreMatchedById()
    {
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var tasks = Enumerable.Range(0, 50)
            .Select(i => client.SendAsync("ping", new { echo = i.ToString() }))
            .ToArray();
        var results = await Task.WhenAll(tasks).WaitAsync(Timeout);

        for (var i = 0; i < results.Length; i++)
        {
            Assert.Equal(i.ToString(), results[i].GetProperty("echo").GetString());
        }
    }

    [Fact]
    public async Task UnknownCommand_ThrowsWorkerException_AndWorkerKeepsRunning()
    {
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var ex = await Assert.ThrowsAsync<WorkerException>(() => client.SendAsync("does_not_exist"));
        Assert.Equal("unknown_command", ex.Code);

        await client.SendAsync("ping").WaitAsync(Timeout);
        Assert.True(client.IsRunning);
    }

    [Fact]
    public async Task Info_ReportsPythonVersion()
    {
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var info = await client.SendAsync("info").WaitAsync(Timeout);
        Assert.Equal(JsonValueKind.String, info.GetProperty("python").ValueKind);
    }

    [Fact]
    public async Task Stop_ExitsProcess()
    {
        var client = new PythonWorkerClient();
        await client.StartAsync();
        var pid = client.WorkerPid!.Value;

        await client.StopAsync();

        Assert.False(client.IsRunning);
        Assert.Throws<ArgumentException>(() => Process.GetProcessById(pid));
        await Assert.ThrowsAsync<WorkerUnavailableException>(() => client.SendAsync("ping"));
        await client.DisposeAsync();
    }

    [Fact]
    public async Task Crash_FailsRequestsWithWorkerUnavailable()
    {
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        // Kill the interpreter itself (not a launcher), as a real crash would.
        Process.GetProcessById(client.WorkerPid!.Value).Kill();

        await Assert.ThrowsAsync<WorkerUnavailableException>(async () =>
        {
            // The first send may race the exit notification; either it fails or the next one does.
            await client.SendAsync("ping").WaitAsync(Timeout);
            await client.SendAsync("ping").WaitAsync(Timeout);
        });
    }

    [Fact]
    public async Task Start_WithMissingPython_ThrowsWorkerUnavailable()
    {
        await using var client = new PythonWorkerClient(new WorkerOptions { PythonExecutable = "definitely-not-python-xyz" });

        await Assert.ThrowsAsync<WorkerUnavailableException>(() => client.StartAsync());
    }

    [Fact]
    public async Task Start_WithMissingScript_ThrowsWorkerUnavailable()
    {
        await using var client = new PythonWorkerClient(new WorkerOptions { ScriptPath = "Z:\\missing\\main.py" });

        await Assert.ThrowsAsync<WorkerUnavailableException>(() => client.StartAsync());
    }
}
