using System.Threading;
using System.Threading.Tasks;

namespace ImageQualityEnhancer.Services;

/// <summary>
/// Lets the scheduler stop starting new work while paused. Work already running is not interrupted.
/// </summary>
public sealed class PauseGate
{
    private readonly object _lock = new();
    private TaskCompletionSource? _paused;

    public bool IsPaused
    {
        get
        {
            lock (_lock)
            {
                return _paused is not null;
            }
        }
    }

    /// <summary>Optional explanation when the job paused itself (e.g. low disk space).</summary>
    public string? Reason { get; private set; }

    public void Pause(string? reason = null)
    {
        lock (_lock)
        {
            Reason = reason;
            _paused ??= new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }
    }

    public void Resume()
    {
        lock (_lock)
        {
            Reason = null;
            _paused?.TrySetResult();
            _paused = null;
        }
    }

    /// <summary>Completes immediately when running; otherwise waits for <see cref="Resume"/> or cancellation.</summary>
    public Task WaitIfPausedAsync(CancellationToken ct)
    {
        Task? wait;
        lock (_lock)
        {
            wait = _paused?.Task;
        }

        return wait is null ? Task.CompletedTask : wait.WaitAsync(ct);
    }
}
