using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services.Docx;

namespace ImageQualityEnhancer.Services;

/// <summary>Outcome of a successful export.</summary>
/// <param name="Destination">The saved document, or the folder holding the saved images.</param>
/// <param name="ImagesReplaced">Approved images now in the output.</param>
/// <param name="NotReplaced">Approved images that could not be written and were left as the original, with reasons.</param>
/// <param name="Files">Files written (one for documents, one per image file for images).</param>
public sealed record ExportResult(
    string Destination,
    int ImagesReplaced,
    IReadOnlyList<string> NotReplaced,
    IReadOnlyList<string> Files,
    string Validation);

public sealed partial class JobOrchestrator
{
    private readonly SemaphoreSlim _exportLock = new(1, 1);

    /// <summary>
    /// Writes a new document (or new image files) containing only the approved enhancements.
    ///
    /// The output is built inside the job workspace, validated against the original, and only then copied to
    /// <paramref name="destination"/> (via a temporary name, renamed at the end). The original is never written.
    /// For image inputs <paramref name="destination"/> is a folder; files get an "_enhanced" suffix.
    /// </summary>
    public async Task<ExportResult> ExportAsync(string destination, string? password = null, CancellationToken ct = default)
    {
        await _exportLock.WaitAsync(ct);
        List<ImageJob> approved;
        lock (_stateLock)
        {
            if (State?.Stage != JobStage.AwaitingReview)
            {
                _exportLock.Release();
                throw new InvalidOperationException("Export is available once processing has finished.");
            }

            approved = ApprovalRules.ToReplace(State.Images).Select(i => i.Clone()).ToList();
            State.Stage = JobStage.Exporting;
        }

        if (password is not null)
        {
            _password = password;
        }

        var sw = Stopwatch.StartNew();
        try
        {
            if (approved.Count == 0)
            {
                throw new JobFailure("nothing_approved", "No enhancement is approved yet, so there is nothing to export.");
            }

            SetStage(JobStage.Exporting, 0, approved.Count, "Preparing the export");
            Report(force: true);
            var result = State!.SourceType switch
            {
                SourceType.Pdf => await ExportPdfAsync(approved, destination, ct),
                SourceType.Docx => await ExportDocxAsync(approved, destination, ct),
                _ => await ExportImagesAsync(approved, destination, ct),
            };

            lock (_stateLock)
            {
                State.LastExport = new ExportRecord(result.Destination, DateTimeOffset.UtcNow, result.ImagesReplaced);
                var timings = State.Performance.StageMilliseconds;
                timings["export"] = sw.Elapsed.TotalMilliseconds;
            }

            await WriteDiagnosticsAsync(ct);
            SetStage(JobStage.Completed, 1, 1, "Saved and validated");
            return result;
        }
        catch (Exception ex) when (ex is not JobFailure && ex is not OperationCanceledException && ToFailure(ex) is { } failure)
        {
            throw failure;
        }
        finally
        {
            lock (_stateLock)
            {
                State!.Stage = JobStage.AwaitingReview; // further review and another export stay possible
            }

            await SaveAsync(CancellationToken.None);
            Report(force: true);
            _exportLock.Release();
        }
    }

    // ------------------------------------------------------------------ PDF

    private async Task<ExportResult> ExportPdfAsync(List<ImageJob> approved, string destination, CancellationToken ct)
    {
        EnsureNotOriginal(destination);
        var output = PrepareExportFile(".pdf");
        var replacements = approved
            .GroupBy(j => j.Location.Xref!.Value)
            .Select(g => new { xref = g.Key, path = Abs(g.First().ResultPath!) })
            .ToList();
        var progress = new Progress<JsonElement>(p =>
        {
            if (p.TryGetProperty("done", out var done) && p.TryGetProperty("total", out var total))
            {
                SetStage(JobStage.Exporting, done.GetInt32(), total.GetInt32() + 1,
                    $"Replacing image {done.GetInt32()} of {total.GetInt32()}");
            }
            else if (p.TryGetProperty("stage", out var stage) && stage.GetString() == "validate")
            {
                SetStage(JobStage.Exporting, replacements.Count, replacements.Count + 1, "Validating the new PDF");
            }

            Report();
        });

        var worker = await WorkerAsync(0, ct);
        var result = await worker.SendAsync("pdf_replace", new
        {
            source = Abs(State!.OriginalCopyPath!),
            output,
            password = _password,
            replacements,
            chunkBytes = 64L * 1024 * 1024,
        }, progress, ct);

        var byXref = approved.ToLookup(j => j.Location.Xref!.Value);
        var notReplaced = result.GetProperty("failed").EnumerateArray()
            .Select(f => $"{Describe(byXref[f.GetProperty("xref").GetInt32()].First())}: {f.GetProperty("message").GetString()}")
            .ToList();
        var validation = result.GetProperty("validation");
        await PublishAsync(output, destination, ct);
        return new ExportResult(destination, result.GetProperty("replaced").GetArrayLength(), notReplaced, [destination],
            $"Checked {validation.GetProperty("streamsCompared").GetInt32():N0} unchanged objects and " +
            $"{validation.GetProperty("pagesRendered").GetInt32():N0} pages; the original is unchanged.");
    }

    // ------------------------------------------------------------------ DOCX

    private async Task<ExportResult> ExportDocxAsync(List<ImageJob> approved, string destination, CancellationToken ct)
    {
        EnsureNotOriginal(destination);
        var output = PrepareExportFile(Path.GetExtension(State!.OriginalCopyPath!));
        var replacements = approved
            .Select(j => new DocxReplacement(j.Location.PartUri!, j.Location.Ordinal!.Value, j.Location.ImageUri!, Abs(j.ResultPath!)))
            .ToList();
        var progress = new Progress<DocxExportProgress>(p =>
        {
            SetStage(JobStage.Exporting, p.Stage == "validate" ? p.Total : p.Done, Math.Max(1, p.Total) + 1,
                p.Stage == "validate" ? "Validating the new document" : "Writing the new document");
            Report();
        });

        var result = await Task.Run(() => DocxExporter.Export(Abs(State.OriginalCopyPath!), output, replacements, progress, ct), ct);
        var byKey = approved.ToDictionary(j => (j.Location.PartUri!, j.Location.Ordinal!.Value));
        var notReplaced = result.Failed
            .Select(f => $"{Describe(byKey[(f.Replacement.PartUri, f.Replacement.Ordinal)])}: {f.Message}")
            .ToList();
        await PublishAsync(output, destination, ct);
        return new ExportResult(destination, result.Applied.Count, notReplaced, [destination],
            $"Checked {result.Validation.EntriesCompared:N0} document parts and {result.Validation.ImagesChecked:N0} image " +
            "placements; text, layout and unapproved images are unchanged.");
    }

    // ------------------------------------------------------------------ image files

    private async Task<ExportResult> ExportImagesAsync(List<ImageJob> approved, string folder, CancellationToken ct)
    {
        if (!Directory.Exists(folder))
        {
            throw new JobFailure("bad_destination", "The chosen folder does not exist.");
        }

        var worker = await WorkerAsync(0, ct);
        var written = new List<string>();
        var notReplaced = new List<string>();
        var files = approved.GroupBy(j => j.Location.SourcePath!).ToList();
        var done = 0;
        foreach (var file in files)
        {
            ct.ThrowIfCancellationRequested();
            List<ImageJob> pages;
            lock (_stateLock)
            {
                pages = State!.Images.Where(i => i.Location.SourcePath == file.Key).OrderBy(i => i.Location.FrameIndex).Select(i => i.Clone()).ToList();
            }

            var lead = file.First();
            var enhancedExtension = Path.GetExtension(lead.ResultPath!);
            string built;
            int expectedFrames;
            if (pages.Count > 1)
            {
                // Multi-page TIFF: approved pages enhanced, every other page exactly as it was.
                var approvedFrames = file.ToDictionary(j => j.Location.FrameIndex ?? 0, j => Abs(j.ResultPath!));
                built = PrepareExportFile(".tif", $"{lead.ResourceId}");
                var frames = pages.Select(p => approvedFrames.TryGetValue(p.Location.FrameIndex ?? 0, out var path)
                    ? (object)new { path }
                    : new { path = file.Key, frame = p.Location.FrameIndex ?? 0 }).ToList();
                await worker.SendAsync("assemble_tiff", new { frames, output = built, source = file.Key }, ct: ct);
                enhancedExtension = Path.GetExtension(file.Key);
                expectedFrames = pages.Count;
            }
            else
            {
                built = Abs(lead.ResultPath!);
                expectedFrames = 1;
            }

            // Validate before publishing: it decodes as the expected format, size and page count.
            var probe = await worker.SendAsync("probe", new { path = built }, ct: ct);
            var expected = lead.OutputPixelSize;
            if (probe.GetProperty("kind").GetString() != "raster"
                || probe.GetProperty("frames").GetInt32() != expectedFrames
                || (pages.Count == 1 && expected is not null
                    && (probe.GetProperty("width").GetInt32() != expected.Width || probe.GetProperty("height").GetInt32() != expected.Height)))
            {
                notReplaced.Add($"{Path.GetFileName(file.Key)}: the enhanced file did not pass validation");
                continue;
            }

            var target = UniqueName(folder, Path.GetFileNameWithoutExtension(file.Key) + "_enhanced" + enhancedExtension);
            EnsureNotOriginal(target);
            await PublishAsync(built, target, ct);
            written.Add(target);
            SetStage(JobStage.Exporting, ++done, files.Count + 1, $"Saved {done} of {files.Count} images");
            Report();
        }

        if (written.Count == 0)
        {
            throw new JobFailure("validation_failed", "None of the enhanced images passed validation; nothing was saved.");
        }

        return new ExportResult(folder, written.Count, notReplaced, written,
            "Each saved image was decoded again and matches its expected format and size; the originals are unchanged.");
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>A fresh file in the workspace's export folder.</summary>
    private string PrepareExportFile(string extension, string name = "output")
    {
        var path = Workspace!.Resolve(WorkspaceFolder.Export, name + extension);
        if (File.Exists(path))
        {
            File.Delete(path);
        }

        return path;
    }

    /// <summary>The user's original files are never overwritten.</summary>
    private void EnsureNotOriginal(string destination)
    {
        var full = Path.GetFullPath(destination);
        if (State!.SourcePaths.Any(s => string.Equals(Path.GetFullPath(s), full, StringComparison.OrdinalIgnoreCase)))
        {
            throw new JobFailure("same_as_original", "Choose a different file name: the original is never overwritten.");
        }
    }

    /// <summary>
    /// Copies a validated output to its final location: first under a temporary name next to the destination,
    /// then renamed, so an interrupted copy never appears as a finished document.
    /// </summary>
    private static async Task PublishAsync(string validated, string destination, CancellationToken ct)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(destination))!;
        var size = new FileInfo(validated).Length;
        var root = Path.GetPathRoot(directory);
        if (root is not null && !root.StartsWith(@"\\", StringComparison.Ordinal)
            && new DriveInfo(root).AvailableFreeSpace < size + 16L * 1024 * 1024)
        {
            throw new JobFailure("disk_full", $"Not enough free space at the destination ({size / 1048576 + 1} MB needed).");
        }

        var temp = Path.Combine(directory, $".{Path.GetFileName(destination)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await using (var from = new FileStream(validated, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true))
            await using (var to = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
            {
                await from.CopyToAsync(to, ct);
                await to.FlushAsync(ct);
            }

            if (new FileInfo(temp).Length != size)
            {
                throw new JobFailure("write_failed", "The document was not copied completely; nothing was saved.");
            }

            File.Move(temp, destination, overwrite: true);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new JobFailure("access_denied", "This location cannot be written to. Choose another folder.", ex);
        }
        finally
        {
            if (File.Exists(temp))
            {
                File.Delete(temp);
            }
        }
    }

    private static string UniqueName(string folder, string fileName)
    {
        var path = Path.Combine(folder, fileName);
        var stem = Path.GetFileNameWithoutExtension(fileName);
        var extension = Path.GetExtension(fileName);
        for (var n = 2; File.Exists(path); n++)
        {
            path = Path.Combine(folder, $"{stem} ({n}){extension}");
        }

        return path;
    }

    private static string Describe(ImageJob job) => job.SourceType switch
    {
        SourceType.Pdf => $"image on page {job.Location.Placements?.FirstOrDefault()?.PageIndex + 1}",
        SourceType.Docx => $"image {job.Location.Ordinal + 1} in {Path.GetFileNameWithoutExtension(job.Location.PartUri)}",
        _ => Path.GetFileName(job.Location.SourcePath ?? job.Id),
    };

    // ------------------------------------------------------------------ diagnostics

    /// <summary>
    /// Technical diagnostics without any document content: no file names, text or pixels. Saved with the job and
    /// shown in the optional diagnostics panel.
    /// </summary>
    public DiagnosticsReport GetDiagnostics()
    {
        lock (_stateLock)
        {
            var state = State!;
            var perf = state.Performance;
            return new DiagnosticsReport(
                SourceType: state.SourceType.ToString(),
                TotalImages: state.Images.Count,
                Enhanced: state.Images.Count(i => i.Status == ProcessingStatus.Completed),
                Skipped: state.Images.Count(i => i.Status == ProcessingStatus.Skipped),
                Failed: state.Images.Count(i => i.Status == ProcessingStatus.Failed),
                SkipReasons: state.Images.Where(i => i.Status == ProcessingStatus.Skipped && i.Reason is not null)
                    .GroupBy(i => i.Reason!).ToDictionary(g => g.Key, g => g.Count()),
                FailureReasons: state.Images.Where(i => i.Status == ProcessingStatus.Failed && i.Reason is not null)
                    .GroupBy(i => i.Reason!).ToDictionary(g => g.Key, g => g.Count()),
                StageSeconds: perf.StageMilliseconds.ToDictionary(kv => kv.Key, kv => Math.Round(kv.Value / 1000, 2)),
                AverageEnhanceSeconds: perf.AverageEnhanceMilliseconds is { } avg ? Math.Round(avg / 1000, 3) : null,
                PeakWorkerMemoryMegabytes: perf.PeakWorkerMemoryBytes / 1048576,
                TempDiskMegabytes: (Workspace is not null && Directory.Exists(Workspace.RootPath) ? Workspace.GetSizeBytes() : perf.TempDiskBytes) / 1048576,
                WorkerCount: _options.WorkerCount,
                WorkerRestarts: perf.WorkerRestarts,
                Batches: perf.BatchCount);
        }
    }

    private async Task WriteDiagnosticsAsync(CancellationToken ct)
    {
        var path = Workspace!.Resolve(WorkspaceFolder.Metadata, "diagnostics.json");
        await File.WriteAllBytesAsync(path, JsonSerializer.SerializeToUtf8Bytes(GetDiagnostics(), JobStateStore.JsonOptions), ct);
    }

    /// <summary>Deletes the job's temporary files once the user is done with it.</summary>
    public async Task FinishAsync()
    {
        await StopWorkersAsync(_options.CancelGracePeriod);
        if (Workspace is not null && Directory.Exists(Workspace.RootPath))
        {
            _workspaces.Delete(Workspace);
        }
    }
}

public sealed record DiagnosticsReport(
    string SourceType,
    int TotalImages,
    int Enhanced,
    int Skipped,
    int Failed,
    IReadOnlyDictionary<string, int> SkipReasons,
    IReadOnlyDictionary<string, int> FailureReasons,
    IReadOnlyDictionary<string, double> StageSeconds,
    double? AverageEnhanceSeconds,
    long PeakWorkerMemoryMegabytes,
    long TempDiskMegabytes,
    int WorkerCount,
    int WorkerRestarts,
    int Batches);
