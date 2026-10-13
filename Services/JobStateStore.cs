using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using ImageQualityEnhancer.Models;

namespace ImageQualityEnhancer.Services;

/// <summary>
/// Saves and loads the job queue (metadata/queue.json). Writes go to a temp file first and are then
/// moved over the old file, so a crash mid-save never leaves a truncated queue behind.
/// </summary>
public static class JobStateStore
{
    private const string FileName = "queue.json";

    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseUpper) },
    };

    public static Task SaveAsync(JobWorkspace workspace, JobState state, CancellationToken ct = default) =>
        SaveAsync(workspace, Serialize(state), ct);

    /// <summary>Serializes in memory, so callers can take a consistent snapshot while holding their own lock.</summary>
    public static byte[] Serialize(JobState state) => JsonSerializer.SerializeToUtf8Bytes(state, JsonOptions);

    public static async Task SaveAsync(JobWorkspace workspace, byte[] json, CancellationToken ct = default)
    {
        var target = workspace.Resolve(WorkspaceFolder.Metadata, FileName);
        var temp = target + ".tmp";

        await using (var stream = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            await stream.WriteAsync(json, ct);
            await stream.FlushAsync(ct);
        }

        File.Move(temp, target, overwrite: true);
    }

    public static async Task<JobState?> LoadAsync(JobWorkspace workspace, CancellationToken ct = default)
    {
        var path = workspace.Resolve(WorkspaceFolder.Metadata, FileName);
        if (!File.Exists(path))
        {
            return null;
        }

        await using var stream = File.OpenRead(path);
        var state = await JsonSerializer.DeserializeAsync<JobState>(stream, JsonOptions, ct);

        // An image that was mid-flight when the app stopped has no trustworthy result; run it again.
        if (state is not null)
        {
            foreach (var image in state.Images)
            {
                if (image.Status == ProcessingStatus.Processing)
                {
                    image.Status = ProcessingStatus.Pending;
                }
            }
        }

        return state;
    }
}
