using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services.Docx;

namespace ImageQualityEnhancer.Services;

public sealed record JobRequest(IReadOnlyList<string> Files, string? Password = null);

/// <summary>What "Try AI" did: pictures tried (each shared picture once), usable AI versions, rejected ones, pictures
/// left unchanged because no suitable, installed AI repair applies, and usable AI versions applied (put in use) because
/// the picture was still to decide.</summary>
public sealed record AiRunSummary(int Tried, int Accepted, int Rejected, int Ineligible, int Applied = 0);

public enum JobOutcomeKind
{
    /// <summary>Every image has been processed; results wait for the user's review.</summary>
    ReadyForReview,

    /// <summary>The user cancelled; the workspace has been cleaned up.</summary>
    Cancelled,

    /// <summary>Stopped from outside (e.g. the app is closing). Progress is saved and the job can be resumed.</summary>
    Interrupted,

    /// <summary>A document-level problem stopped the job (see ErrorCode/Message). Nothing was written.</summary>
    Failed,

    /// <summary>The user finished with the job; its temporary files have been removed.</summary>
    Finished,
}

public sealed record JobOutcome(JobOutcomeKind Kind, string? ErrorCode = null, string? Message = null);

/// <summary>A document-level failure with a stable code and a plain-language message.</summary>
public sealed class JobFailure(string code, string message, Exception? inner = null) : Exception(message, inner)
{
    public string Code { get; } = code;
}

/// <summary>
/// Runs one document-processing job off the UI thread: copy -> scan -> extract -> enhance (in memory-sized batches)
/// -> ready for review. The queue is saved after every stage and batch, so a job interrupted at any point resumes
/// where it stopped. A failed image never stops the job; a document-level problem stops it with an explanation.
/// </summary>
public sealed partial class JobOrchestrator : IAsyncDisposable
{
    private const double ReportInterval = 100; // ms between throttled progress events

    private readonly WorkspaceManager _workspaces;
    private readonly ProcessingOptions _options;
    private readonly Func<PythonWorkerClient> _workerFactory;
    private readonly PauseGate _gate = new();
    private readonly EtaEstimator _eta = new();
    private readonly object _stateLock = new();
    private readonly SemaphoreSlim _saveLock = new(1, 1);
    private readonly SemaphoreSlim _reenhanceLock = new(1, 1);
    private readonly Stopwatch _sinceReport = Stopwatch.StartNew();
    private PythonWorkerClient?[] _workers = [];
    private CancellationTokenSource? _cts;
    private Task<JobOutcome>? _run;
    private volatile bool _cancelRequested;
    private string? _password;
    private (JobStage Stage, int Done, int Total, string Operation) _stage = (JobStage.Created, 0, 0, "");

    public JobOrchestrator(WorkspaceManager workspaces, ProcessingOptions? options = null,
        Func<PythonWorkerClient>? workerFactory = null)
    {
        _workspaces = workspaces;
        _options = options ?? new ProcessingOptions();
        _workerFactory = workerFactory ?? (() => new PythonWorkerClient());
    }

    public JobState? State { get; private set; }
    public JobWorkspace? Workspace { get; private set; }
    public bool IsPaused => _gate.IsPaused;

    /// <summary>Raised on a background thread. Marshal to the UI thread (Dispatcher) before touching controls.</summary>
    public event Action<JobProgress>? ProgressChanged;

    /// <summary>Raised on a background thread with a snapshot of an image whose state changed.</summary>
    public event Action<ImageJob>? ImageUpdated;

    // ------------------------------------------------------------------ public control

    public Task<JobOutcome> RunAsync(JobRequest request, CancellationToken ct = default)
    {
        if (_run is not null)
        {
            throw new InvalidOperationException("This orchestrator already ran a job.");
        }

        _password = request.Password;
        _run = Task.Run(async () =>
        {
            SourceType type;
            try
            {
                type = DetectSourceType(request.Files);
            }
            catch (JobFailure ex)
            {
                return new JobOutcome(JobOutcomeKind.Failed, ex.Code, ex.Message);
            }

            Workspace = _workspaces.Create();
            State = new JobState
            {
                JobId = Workspace.JobId,
                SourceType = type,
                SourceFileName = request.Files.Count == 1 ? Path.GetFileName(request.Files[0]) : $"{request.Files.Count} images",
                SourcePaths = request.Files.Select(Path.GetFullPath).ToList(),
            };
            return await ExecuteAsync(ct);
        }, CancellationToken.None);
        return _run;
    }

    /// <summary>Continues a job from its saved queue (e.g. after the app was closed). Finished images are kept.</summary>
    public Task<JobOutcome> ResumeAsync(JobWorkspace workspace, string? password = null, CancellationToken ct = default)
    {
        if (_run is not null)
        {
            throw new InvalidOperationException("This orchestrator already ran a job.");
        }

        _password = password;
        _run = Task.Run(async () =>
        {
            var state = await JobStateStore.LoadAsync(workspace, ct)
                        ?? throw new InvalidOperationException("The job has no saved queue to resume.");
            Workspace = workspace;
            State = state;
            if (state.Stage is JobStage.Failed or JobStage.Cancelled)
            {
                return new JobOutcome(JobOutcomeKind.Failed, state.ErrorCode, state.Error ?? "This job cannot be resumed.");
            }

            return await ExecuteAsync(ct);
        }, CancellationToken.None);
        return _run;
    }

    /// <summary>Stops scheduling new images. The image being processed finishes.</summary>
    public void Pause()
    {
        _gate.Pause();
        Report(force: true);
    }

    public void Resume()
    {
        _gate.Resume();
        Report(force: true);
    }

    /// <summary>
    /// Stops the job: no new images are started, the current one gets a short grace period, then the worker is
    /// stopped. Unfinished images are marked CANCELLED and the workspace is removed (unless asked to keep it).
    /// </summary>
    public async Task CancelAsync(bool deleteWorkspace = true)
    {
        _cancelRequested = true;
        _gate.Resume();
        _cts?.Cancel();
        await StopWorkersAsync(_options.CancelGracePeriod);
        if (_run is not null)
        {
            await _run.ContinueWith(_ => { }, TaskScheduler.Default);
        }

        if (State is not null && Workspace is not null)
        {
            lock (_stateLock)
            {
                foreach (var image in State.Images.Where(i => i.Status is ProcessingStatus.Pending or ProcessingStatus.Processing))
                {
                    image.Status = ProcessingStatus.Cancelled;
                }

                State.Stage = JobStage.Cancelled;
            }

            if (deleteWorkspace)
            {
                _workspaces.Delete(Workspace);
            }
            else
            {
                await SaveAsync(CancellationToken.None);
            }
        }
    }

    // ------------------------------------------------------------------ review actions (after processing)

    public IReadOnlyList<ImageJob> SnapshotImages()
    {
        lock (_stateLock)
        {
            return State?.Images.Select(i => i.Clone()).ToList() ?? [];
        }
    }

    /// <summary>Approves or keeps the original for the given images. Images that cannot be approved are left as they are.</summary>
    public Task<int> SetApprovalAsync(IReadOnlyCollection<string> ids, ApprovalStatus approval) =>
        ChangeApprovalsAsync(images =>
        {
            var wanted = ids.ToHashSet();
            return images.Where(i => wanted.Contains(i.Id) && i.Approval != approval && ApprovalRules.Set(i, approval)).ToList();
        });

    public Task<int> ApproveAllEligibleAsync() =>
        ChangeApprovalsAsync(images =>
        {
            var targets = images.Where(i => i.Approval == ApprovalStatus.Undecided && ApprovalRules.IsEligibleForBulkApproval(i)).ToList();
            ApprovalRules.ApproveAllEligible(targets);
            return targets;
        });

    public Task<int> RejectAllAsync() =>
        ChangeApprovalsAsync(images =>
        {
            var targets = images.Where(i => i.Approval != ApprovalStatus.KeepOriginal).ToList();
            ApprovalRules.RejectAll(targets);
            return targets;
        });

    private async Task<int> ChangeApprovalsAsync(Func<List<ImageJob>, List<ImageJob>> change)
    {
        List<ImageJob> snapshots;
        lock (_stateLock)
        {
            snapshots = change(State!.Images).Select(i => i.Clone()).ToList();
        }

        await SaveAsync(CancellationToken.None);
        foreach (var snapshot in snapshots)
        {
            ImageUpdated?.Invoke(snapshot);
        }

        return snapshots.Count;
    }

    /// <summary>Whether an image can be enhanced again with different settings.</summary>
    public static bool CanReenhance(ImageJob image) =>
        image.Status == ProcessingStatus.Completed
        || (image.Status == ProcessingStatus.Skipped && image.Reason is "already_good" or "disabled");

    /// <summary>
    /// Enhances one image again with new settings (e.g. manual adjustments) once processing has finished.
    /// Applies to every place the same image appears. The previous decision is cleared, since the result changed.
    /// </summary>
    public async Task ReenhanceAsync(string imageId, EnhancementSettings settings, CancellationToken ct = default)
    {
        await _reenhanceLock.WaitAsync(ct);
        try
        {
            List<ImageJob> group;
            lock (_stateLock)
            {
                if (State?.Stage != JobStage.AwaitingReview)
                {
                    throw new InvalidOperationException("Images can be adjusted once processing has finished.");
                }

                var image = State.Images.Single(i => i.Id == imageId);
                if (!CanReenhance(image))
                {
                    throw new InvalidOperationException("This image cannot be enhanced.");
                }

                group = State.Images.Where(i => i.ResourceRef == image.ResourceRef && CanReenhance(i)).ToList();
                foreach (var job in group)
                {
                    job.Settings = settings;
                    job.Status = ProcessingStatus.Pending;
                    job.Approval = ApprovalStatus.Undecided;
                    job.Reason = null;
                    job.Error = null;
                    job.NeedsReview = false;
                    job.ReviewReasons = [];
                    job.Operations = [];
                    job.Notes = [];
                    job.Attempts = 0;
                    job.Ai = null;
                    job.UseAiVersion = false;
                }
            }

            var budget = BatchPlanner.Budget(await AvailableMemoryAsync(ct), _options.AvailableRamFraction, _options.MaxImageMemoryBytes, 1);
            var limit = BatchPlanner.PerImageLimit(budget, _options.MaxImageMemoryBytes, 1);
            while (group[0].Status == ProcessingStatus.Pending)
            {
                await EnhanceGroupAsync(group, 0, limit, ct); // retries once if the worker crashes
            }

            await SaveAsync(ct);
            SetStage(JobStage.AwaitingReview, 0, 0, "Ready for review");
            Report(force: true);
        }
        finally
        {
            _reenhanceLock.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts?.Cancel();
        await StopWorkersAsync(_options.CancelGracePeriod);
        _cts?.Dispose();
        _saveLock.Dispose();
        _reenhanceLock.Dispose();
        _exportLock.Dispose();
    }

    // ------------------------------------------------------------------ pipeline

    private async Task<JobOutcome> ExecuteAsync(CancellationToken external)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(external);
        var ct = _cts.Token;
        lock (_stateLock)
        {
            State!.ErrorCode = null; // e.g. a password problem from an earlier attempt
            State.Error = null;
        }

        try
        {
            await PipelineAsync(ct);
            return new JobOutcome(JobOutcomeKind.ReadyForReview);
        }
        catch (OperationCanceledException) when (_cancelRequested)
        {
            return new JobOutcome(JobOutcomeKind.Cancelled);
        }
        catch (Exception ex) when (_cancelRequested && ex is WorkerUnavailableException)
        {
            return new JobOutcome(JobOutcomeKind.Cancelled);
        }
        catch (OperationCanceledException)
        {
            lock (_stateLock)
            {
                foreach (var image in State!.Images.Where(i => i.Status == ProcessingStatus.Processing))
                {
                    image.Status = ProcessingStatus.Pending;
                }
            }

            await SaveAsync(CancellationToken.None);
            return new JobOutcome(JobOutcomeKind.Interrupted);
        }
        catch (Exception ex) when (ToFailure(ex) is { } failure)
        {
            lock (_stateLock)
            {
                // A missing password is not final: the job stays resumable once the user enters it.
                // (The password itself is never saved.)
                if (failure.Code is not ("password_required" or "wrong_password"))
                {
                    State!.Stage = JobStage.Failed;
                }

                State!.ErrorCode = failure.Code;
                State.Error = failure.Message;
            }

            await SaveAsync(CancellationToken.None);
            Report(force: true);
            return new JobOutcome(JobOutcomeKind.Failed, failure.Code, failure.Message);
        }
        finally
        {
            if (!_cancelRequested)
            {
                await StopWorkersAsync(_options.CancelGracePeriod);
            }
        }
    }

    private async Task PipelineAsync(CancellationToken ct)
    {
        var state = State!;
        if (state.Stage == JobStage.Created)
        {
            await TimedAsync("copy", () => CopyOriginalAsync(ct));
            await AdvanceAsync(JobStage.Scanning, ct);
        }

        if (state.Stage == JobStage.Scanning)
        {
            await TimedAsync("scan", () => ScanAsync(ct));
            await AdvanceAsync(JobStage.Extracting, ct);
        }

        if (state.Stage == JobStage.Extracting)
        {
            await TimedAsync("extract", () => ExtractAsync(ct));
            await AdvanceAsync(JobStage.Classifying, ct);
        }

        if (state.Stage == JobStage.Classifying)
        {
            await TimedAsync("classify", () => ClassifyAsync(ct));
            await AdvanceAsync(JobStage.Enhancing, ct);
        }

        if (state.Stage == JobStage.Enhancing)
        {
            // Jobs saved before classification existed: their pictures are classified before they are enhanced.
            await ClassifyAsync(ct);
            await TimedAsync("enhance", () => EnhanceAsync(ct));
            lock (_stateLock)
            {
                state.Performance.TempDiskBytes = Workspace!.GetSizeBytes();
            }

            await AdvanceAsync(JobStage.AwaitingReview, ct);
        }
    }

    private async Task AdvanceAsync(JobStage next, CancellationToken ct)
    {
        lock (_stateLock)
        {
            State!.Stage = next;
        }

        SetStage(next, 0, 0, next == JobStage.AwaitingReview ? "Ready for review" : "");
        await SaveAsync(ct);
        Report(force: true);
    }

    private async Task TimedAsync(string name, Func<Task> stage)
    {
        var sw = Stopwatch.StartNew();
        await stage();
        lock (_stateLock)
        {
            var timings = State!.Performance.StageMilliseconds;
            timings[name] = timings.GetValueOrDefault(name) + sw.Elapsed.TotalMilliseconds;
        }
    }

    // ------------------------------------------------------------------ copy

    private async Task CopyOriginalAsync(CancellationToken ct)
    {
        var state = State!;
        var totalInput = state.SourcePaths.Sum(p => new FileInfo(p).Length);
        // Original copy + extracted + enhanced images + exported document, plus headroom.
        var needed = (state.SourceType == SourceType.Image ? 2 : 3) * totalInput + _options.MinFreeDiskBytes;
        var free = _workspaces.GetAvailableFreeSpace();
        if (free < needed)
        {
            throw new JobFailure("disk_full",
                $"Not enough free disk space: about {needed / 1048576} MB is needed and {free / 1048576} MB is available.");
        }

        if (state.SourceType == SourceType.Image)
        {
            return; // image files are read in place (never written); no copy needed
        }

        var source = state.SourcePaths[0];
        var target = Workspace!.Resolve(WorkspaceFolder.Original, "source" + Path.GetExtension(source).ToLowerInvariant());
        SetStage(JobStage.Created, 0, 1, "Copying the original document");

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        await using (var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true))
        await using (var output = new FileStream(target, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
        {
            var buffer = new byte[1 << 16];
            int read;
            while ((read = await input.ReadAsync(buffer, ct)) > 0)
            {
                hash.AppendData(buffer, 0, read);
                await output.WriteAsync(buffer.AsMemory(0, read), ct);
            }
        }

        lock (_stateLock)
        {
            state.OriginalCopyPath = Workspace.ToRelative(target);
            state.SourceSha256 = Convert.ToHexString(hash.GetHashAndReset());
        }
    }

    // ------------------------------------------------------------------ scan

    private async Task ScanAsync(CancellationToken ct)
    {
        var images = State!.SourceType switch
        {
            SourceType.Pdf => await ScanPdfAsync(ct),
            SourceType.Docx => await ScanDocxAsync(ct),
            _ => await ScanImageFilesAsync(ct),
        };

        lock (_stateLock)
        {
            State.Images = images;
        }

        foreach (var image in images)
        {
            ImageUpdated?.Invoke(image.Clone());
        }
    }

    private async Task<List<ImageJob>> ScanPdfAsync(CancellationToken ct)
    {
        var path = Abs(State!.OriginalCopyPath!);
        var worker = await WorkerAsync(0, ct);
        var images = new Dictionary<int, JsonElement>();
        var placements = new Dictionary<int, List<PdfPlacement>>();
        var inline = new List<PdfPlacement>();
        int start = 0, pageCount = int.MaxValue;
        SetStage(JobStage.Scanning, 0, 0, "Scanning pages");

        while (start < pageCount)
        {
            await _gate.WaitIfPausedAsync(ct);
            var end = start + _options.PdfPagesPerScanRequest;
            var result = await worker.SendAsync("pdf_scan", new { path, password = _password, pageStart = start, pageEnd = end }, ct: ct);
            pageCount = result.GetProperty("pageCount").GetInt32();
            if (result.TryGetProperty("signed", out var signed) && signed.GetBoolean())
            {
                lock (_stateLock)
                {
                    State.DigitallySigned = true;
                }
            }

            foreach (var image in result.GetProperty("images").EnumerateArray())
            {
                images.TryAdd(image.GetProperty("xref").GetInt32(), image.Clone());
            }

            foreach (var occ in result.GetProperty("occurrences").EnumerateArray())
            {
                var placement = new PdfPlacement(
                    occ.GetProperty("page").GetInt32(),
                    occ.GetProperty("rect").EnumerateArray().Select(v => v.GetDouble()).ToArray(),
                    occ.GetProperty("inForm").GetBoolean(),
                    occ.GetProperty("fullPage").GetBoolean());
                var xref = occ.GetProperty("xref").GetInt32();
                if (xref == 0)
                {
                    inline.Add(placement);
                }
                else
                {
                    (placements.TryGetValue(xref, out var list) ? list : placements[xref] = []).Add(placement);
                }
            }

            start = end;
            SetStage(JobStage.Scanning, Math.Min(start, pageCount), pageCount, $"Scanning page {Math.Min(start, pageCount)} of {pageCount}");
            Report();
        }

        var jobs = new List<ImageJob>();
        var resources = new Dictionary<string, string>();
        foreach (var (xref, places) in placements.OrderBy(p => p.Value[0].PageIndex).ThenBy(p => p.Key))
        {
            if (!images.TryGetValue(xref, out var info))
            {
                continue;
            }

            var key = info.GetProperty("resourceKey").GetString()!;
            var rect = places[0].Rect;
            var job = new ImageJob
            {
                Id = NextId(jobs),
                DocumentId = State.JobId,
                SourceType = SourceType.Pdf,
                Location = new ImageLocation { Xref = xref, Placements = places },
                ResourceRef = key,
                ResourceId = ResourceIdFor(resources, key),
                PixelSize = new PixelSize(info.GetProperty("width").GetInt32(), info.GetProperty("height").GetInt32()),
                DisplaySize = new DisplaySize(rect[2] - rect[0], rect[3] - rect[1]),
                OutputFormat = info.TryGetProperty("outputFormat", out var f) ? f.GetString() : null,
                Settings = _options.DefaultSettings,
            };
            if (!info.GetProperty("eligible").GetBoolean())
            {
                Skip(job, info.GetProperty("skipReason").GetString()!, info.GetProperty("message").GetString()!);
            }

            jobs.Add(job);
        }

        foreach (var place in inline)
        {
            var job = new ImageJob
            {
                Id = NextId(jobs),
                DocumentId = State.JobId,
                SourceType = SourceType.Pdf,
                Location = new ImageLocation { Placements = [place] },
                ResourceRef = $"inline:{jobs.Count}",
                ResourceId = ResourceIdFor(resources, $"inline:{jobs.Count}"),
                DisplaySize = new DisplaySize(place.Rect[2] - place.Rect[0], place.Rect[3] - place.Rect[1]),
            };
            Skip(job, "inline_image", "image is embedded directly in the page content and is kept unchanged");
            jobs.Add(job);
        }

        return jobs;
    }

    private async Task<List<ImageJob>> ScanDocxAsync(CancellationToken ct)
    {
        var path = Abs(State!.OriginalCopyPath!);
        SetStage(JobStage.Scanning, 0, 0, "Scanning the document");
        Report(force: true);
        var scan = await Task.Run(() => DocxScanner.Scan(path, ct: ct), ct);
        var formats = scan.Images.ToDictionary(i => i.Uri, i => i.OutputFormat, StringComparer.OrdinalIgnoreCase);

        var jobs = new List<ImageJob>();
        var resources = new Dictionary<string, string>();
        foreach (var occ in scan.Occurrences)
        {
            var key = occ.ImageUri ?? $"missing:{occ.PartUri}:{occ.Ref.Ordinal}";
            var job = new ImageJob
            {
                Id = NextId(jobs),
                DocumentId = State.JobId,
                SourceType = SourceType.Docx,
                Location = new ImageLocation
                {
                    PartUri = occ.PartUri,
                    Ordinal = occ.Ref.Ordinal,
                    DrawingId = occ.Ref.DrawingId,
                    ImageUri = occ.ImageUri,
                    InTable = occ.Ref.InTable,
                    Anchored = occ.Ref.Anchored,
                },
                ResourceRef = key,
                ResourceId = ResourceIdFor(resources, key),
                DisplaySize = occ.Ref.ExtentCx is { } cx && occ.Ref.ExtentCy is { } cy ? new DisplaySize(cx / 12700.0, cy / 12700.0) : null,
                OutputFormat = occ.ImageUri is null ? null : formats.GetValueOrDefault(occ.ImageUri),
                Settings = _options.DefaultSettings,
            };
            if (occ.SkipReason is { } reason)
            {
                Skip(job, reason, DocxSkipMessage(reason));
            }

            jobs.Add(job);
        }

        return jobs;
    }

    private async Task<List<ImageJob>> ScanImageFilesAsync(CancellationToken ct)
    {
        var worker = await WorkerAsync(0, ct);
        var files = State!.SourcePaths;
        var jobs = new List<ImageJob>();
        var resources = new Dictionary<string, string>();
        for (var i = 0; i < files.Count; i++)
        {
            await _gate.WaitIfPausedAsync(ct);
            var file = files[i];
            SetStage(JobStage.Scanning, i, files.Count, $"Reading {Path.GetFileName(file)}");
            Report();

            JsonElement probe;
            try
            {
                probe = await worker.SendAsync("probe", new { path = file }, ct: ct);
            }
            catch (WorkerException ex)
            {
                var failed = NewImageFileJob(jobs, resources, file, 0, null);
                failed.Status = ProcessingStatus.Failed;
                failed.Reason = ex.Code;
                failed.Error = ex.Message;
                jobs.Add(failed);
                continue;
            }

            var frames = probe.GetProperty("frames").GetInt32();
            var kind = probe.GetProperty("kind").GetString();
            for (var frame = 0; frame < (kind == "raster" && probe.GetProperty("format").GetString() == "tiff" ? frames : 1); frame++)
            {
                var job = NewImageFileJob(jobs, resources, file, frame, probe);
                switch (kind)
                {
                    case "vector":
                        Skip(job, "vector", "vector graphics are kept unchanged");
                        break;
                    case "unsupported":
                        Skip(job, "unsupported_format", $"{probe.GetProperty("format").GetString()?.ToUpperInvariant()} images cannot be read");
                        break;
                    case "unknown":
                        job.Status = ProcessingStatus.Failed;
                        job.Reason = "unsupported_format";
                        job.Error = "the file is not a recognised image";
                        break;
                }

                jobs.Add(job);
            }
        }

        return jobs;
    }

    private ImageJob NewImageFileJob(List<ImageJob> jobs, Dictionary<string, string> resources, string file, int frame, JsonElement? probe)
    {
        var key = $"{file}#{frame}";
        int? width = probe?.GetProperty("width").ValueKind == JsonValueKind.Number ? probe.Value.GetProperty("width").GetInt32() : null;
        int? height = probe?.GetProperty("height").ValueKind == JsonValueKind.Number ? probe.Value.GetProperty("height").GetInt32() : null;
        return new ImageJob
        {
            Id = NextId(jobs),
            DocumentId = State!.JobId,
            SourceType = SourceType.Image,
            Location = new ImageLocation { SourcePath = file, FrameIndex = frame },
            ResourceRef = key,
            ResourceId = ResourceIdFor(resources, key),
            PixelSize = width is { } w && height is { } h ? new PixelSize(w, h) : null,
            EstimatedMemoryBytes = probe?.TryGetProperty("estimatedMemoryBytes", out var m) == true ? m.GetInt64() : null,
            OutputFormat = probe?.GetProperty("outputFormat").GetString(),
            Settings = _options.DefaultSettings,
        };
    }

    // ------------------------------------------------------------------ extract

    private async Task ExtractAsync(CancellationToken ct)
    {
        List<List<ImageJob>> todo;
        lock (_stateLock)
        {
            todo = State!.Images
                .Where(i => i.Status == ProcessingStatus.Pending && State.SourceType != SourceType.Image)
                .GroupBy(i => i.ResourceRef)
                .Select(g => g.ToList())
                .Where(g => g[0].ExtractedPath is null || !File.Exists(Abs(g[0].ExtractedPath!)))
                .ToList();
        }

        var done = 0;
        SetStage(JobStage.Extracting, 0, todo.Count, "Extracting images");
        Report(force: true);
        if (State!.SourceType == SourceType.Pdf)
        {
            var worker = await WorkerAsync(0, ct);
            foreach (var chunk in todo.Chunk(_options.ImagesPerExtractRequest))
            {
                await _gate.WaitIfPausedAsync(ct);
                var byXref = chunk.ToDictionary(g => g[0].Location.Xref!.Value);
                var result = await worker.SendAsync("pdf_extract", new
                {
                    path = Abs(State.OriginalCopyPath!),
                    password = _password,
                    xrefs = byXref.Keys,
                    outputDir = Workspace!.GetFolder(WorkspaceFolder.Extracted),
                    chunkBytes = 64L * 1024 * 1024,
                }, ct: ct);
                foreach (var item in result.GetProperty("images").EnumerateArray())
                {
                    var group = byXref[item.GetProperty("xref").GetInt32()];
                    if (item.TryGetProperty("error", out var error))
                    {
                        Fail(group, error.GetProperty("code").GetString()!, error.GetProperty("message").GetString()!);
                    }
                    else
                    {
                        var relative = Workspace.ToRelative(item.GetProperty("path").GetString()!);
                        var estimate = item.GetProperty("estimatedMemoryBytes").GetInt64();
                        Update(group, job =>
                        {
                            job.ExtractedPath = relative;
                            job.EstimatedMemoryBytes = estimate;
                        });
                    }

                    SetStage(JobStage.Extracting, ++done, todo.Count, $"Extracting image {done} of {todo.Count}");
                    Report();
                }
            }
        }
        else if (State.SourceType == SourceType.Docx)
        {
            var worker = await WorkerAsync(0, ct);
            var source = Abs(State.OriginalCopyPath!);
            foreach (var group in todo)
            {
                await _gate.WaitIfPausedAsync(ct);
                var imageUri = group[0].Location.ImageUri!;
                var target = Workspace!.Resolve(WorkspaceFolder.Extracted, group[0].ResourceId + SafeExtension(imageUri));
                try
                {
                    await Task.Run(() => DocxScanner.ExtractImage(source, imageUri, target), ct);
                    var probe = await worker.SendAsync("probe", new { path = target }, ct: ct);
                    var relative = Workspace.ToRelative(target);
                    var size = probe.GetProperty("width").ValueKind == JsonValueKind.Number
                        ? new PixelSize(probe.GetProperty("width").GetInt32(), probe.GetProperty("height").GetInt32())
                        : null;
                    var estimate = probe.TryGetProperty("estimatedMemoryBytes", out var m) ? m.GetInt64() : (long?)null;
                    Update(group, job =>
                    {
                        job.ExtractedPath = relative;
                        job.PixelSize = size;
                        job.EstimatedMemoryBytes = estimate;
                    });
                }
                catch (Exception ex) when (ex is DocxException or WorkerException or IOException or InvalidDataException)
                {
                    Fail(group, ex is DocxException d ? d.Code : ex is WorkerException w ? w.Code : "extract_failed", ex.Message);
                }

                SetStage(JobStage.Extracting, ++done, todo.Count, $"Extracting image {done} of {todo.Count}");
                Report();
            }
        }
    }

    // ------------------------------------------------------------------ classify

    /// <summary>
    /// Sorts every extracted picture (and the panels of composite pictures) into chemical diagrams, flowcharts and
    /// graphs, tables and other images with DocumentFigureClassifier, from the extracted file as it is (the classifier
    /// never extracts). Pictures classified as chemical get MolScribe's second check. Each picture shared by several
    /// places is classified once.
    /// </summary>
    private async Task ClassifyAsync(CancellationToken ct)
    {
        List<List<ImageJob>> todo;
        lock (_stateLock)
        {
            todo = State!.Images
                .Where(i => i.Classification is null && i.Status is ProcessingStatus.Pending or ProcessingStatus.Processing)
                .GroupBy(i => i.ResourceRef)
                .Select(g => g.ToList())
                .Where(g => Input(g[0]) is { } input && File.Exists(input))
                .ToList();
        }

        if (todo.Count == 0)
        {
            return;
        }

        var done = 0;
        SetStage(JobStage.Classifying, 0, todo.Count, "Classifying images");
        Report(force: true);
        var worker = await WorkerAsync(0, ct);
        foreach (var group in todo)
        {
            await _gate.WaitIfPausedAsync(ct);
            var lead = group[0];
            try
            {
                var result = await worker.SendAsync("classify", new
                {
                    input = Input(lead),
                    frame = lead.Location.FrameIndex ?? 0,
                    panels = true,
                }, ct: ct);
                if (result.TryGetProperty("available", out var available) && available.ValueKind == JsonValueKind.False)
                {
                    lock (_stateLock)
                    {
                        State!.ClassifierUnavailable = result.TryGetProperty("reason", out var r) ? r.GetString() : "not installed";
                    }

                    break; // nothing is classified; every picture is listed with the other images
                }

                var classification = ParseClassification(result);
                Update(group, job => job.Classification = classification);
            }
            catch (WorkerException ex)
            {
                // A picture that cannot be read is listed with the other images; enhancement reports the problem.
                Update(group, job => job.Classification = new ImageClassification(ImageCategory.Other, null, null, false, [], null,
                    null, "not classified: " + ex.Message));
            }

            SetStage(JobStage.Classifying, ++done, todo.Count, $"Classifying image {done} of {todo.Count}");
            Report();
        }

        await SaveAsync(ct);
    }

    /// <summary>The picture a job's pixels are read from: the extracted copy, or the user's image file (read only).</summary>
    private string? Input(ImageJob job) => job.ExtractedPath is not null ? Abs(job.ExtractedPath) : job.Location.SourcePath;

    /// <summary>Worker group names.</summary>
    internal static string CategoryKey(ImageCategory category) => category switch
    {
        ImageCategory.Chemical => "chemical",
        ImageCategory.FlowchartOrGraph => "flowchart_graph",
        ImageCategory.Table => "table",
        ImageCategory.Other => "other",
        _ => "unclassified",
    };

    internal static ImageCategory CategoryFrom(string? key) => key switch
    {
        "chemical" => ImageCategory.Chemical,
        "flowchart_graph" => ImageCategory.FlowchartOrGraph,
        "table" => ImageCategory.Table,
        "other" => ImageCategory.Other,
        _ => ImageCategory.Unclassified,
    };

    /// <summary>
    /// What the worker is told about a picture before enhancing it: its category and its panels' (the user's choice
    /// replaces them all). Unclassified pictures are never treated as chemical.
    /// </summary>
    internal static object ClassificationRequest(ImageJob job)
    {
        if (job.ManualCategory is { } manual)
        {
            return new { category = CategoryKey(manual), panels = Array.Empty<object>() };
        }

        return new
        {
            category = CategoryKey(job.Classification?.Category ?? ImageCategory.Unclassified),
            panels = (job.Classification?.Panels ?? []).Select(p => new { box = p.Box, category = CategoryKey(p.Category) }).ToArray(),
        };
    }

    internal static ChemicalCheck? ParseChemicalCheck(JsonElement e)
    {
        if (e.ValueKind != JsonValueKind.Object || !e.TryGetProperty("status", out var status))
        {
            return null;
        }

        return new ChemicalCheck(status.GetString() ?? "uncertain",
            e.TryGetProperty("atoms", out var a) && a.ValueKind == JsonValueKind.Number ? a.GetInt32() : 0,
            e.TryGetProperty("bonds", out var b) && b.ValueKind == JsonValueKind.Number ? b.GetInt32() : 0,
            e.TryGetProperty("confidence", out var c) && c.ValueKind == JsonValueKind.Number ? c.GetDouble() : 0,
            e.TryGetProperty("formula", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString() : null,
            e.TryGetProperty("reasons", out var r) && r.ValueKind == JsonValueKind.Array
                ? r.EnumerateArray().Select(x => x.GetString() ?? "").Where(x => x.Length > 0).ToList() : []);
    }

    internal static ImageClassification ParseClassification(JsonElement r)
    {
        static double? Number(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
        static string? Text(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        static bool Flag(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.True;

        var panels = new List<ClassifiedPanel>();
        if (r.TryGetProperty("panels", out var ps) && ps.ValueKind == JsonValueKind.Array)
        {
            foreach (var p in ps.EnumerateArray())
            {
                var box = p.TryGetProperty("box", out var bx) && bx.ValueKind == JsonValueKind.Array
                    ? bx.EnumerateArray().Select(v => v.GetDouble()).ToArray() : [];
                panels.Add(new ClassifiedPanel(p.TryGetProperty("panel", out var n) ? n.GetInt32() : panels.Count + 1, box,
                    CategoryFrom(Text(p, "category")), Number(p, "confidence") ?? 0, Text(p, "label"), Flag(p, "uncertain"),
                    p.TryGetProperty("chemicalCheck", out var pc) ? ParseChemicalCheck(pc) : null));
            }
        }

        return new ImageClassification(CategoryFrom(Text(r, "category")), Number(r, "confidence"), Text(r, "label"),
            Flag(r, "uncertain"), panels, r.TryGetProperty("chemicalCheck", out var cc) ? ParseChemicalCheck(cc) : null,
            Text(r, "model"), Text(r, "note") ?? Text(r, "panelsNote"));
    }

    /// <summary>
    /// The user's category for a picture (null: back to the classifier's suggestion), for every place the same
    /// picture appears. Marking a picture as a chemical diagram runs MolScribe's check on it. When redraw is switched
    /// on and the change decides whether the picture may be redrawn, a finished picture is enhanced again (its
    /// decision is then cleared, as for any new result). A chemical redraw is never approved automatically.
    /// </summary>
    public async Task SetCategoryAsync(string imageId, ImageCategory? category, CancellationToken ct = default)
    {
        List<ImageJob> group;
        bool wasChemical;
        lock (_stateLock)
        {
            var image = State!.Images.Single(i => i.Id == imageId);
            group = State.Images.Where(i => i.ResourceRef == image.ResourceRef).ToList();
            wasChemical = image.IsChemicalCategory;
        }

        ChemicalCheck? check = null;
        if (category == ImageCategory.Chemical && Input(group[0]) is { } input && File.Exists(input))
        {
            try
            {
                var result = await (await WorkerAsync(0, ct)).SendAsync("chemical_check",
                    new { input, frame = group[0].Location.FrameIndex ?? 0 }, ct: ct);
                check = ParseChemicalCheck(result);
            }
            catch (WorkerException ex)
            {
                check = new ChemicalCheck("uncertain", 0, 0, 0, null, [ex.Message]);
            }
        }

        Update(group, job =>
        {
            job.ManualCategory = category;
            job.ManualChemicalCheck = check;
        });
        await SaveAsync(ct);

        ImageJob lead;
        bool isChemical;
        lock (_stateLock)
        {
            lead = group[0];
            isChemical = lead.IsChemicalCategory;
        }

        if (lead.Settings.Redraw && wasChemical != isChemical && State!.Stage == JobStage.AwaitingReview && CanReenhance(lead))
        {
            await ReenhanceAsync(lead.Id, lead.Settings, ct);
        }
    }

    // ------------------------------------------------------------------ optional AI enhancement

    internal static List<AiSuggestion> ParseSuggestions(JsonElement result)
    {
        var found = new List<AiSuggestion>();
        if (result.TryGetProperty("aiSuggestions", out var list) && list.ValueKind == JsonValueKind.Array)
        {
            foreach (var s in list.EnumerateArray())
            {
                found.Add(new AiSuggestion(s.GetProperty("kind").GetString() ?? "", s.GetProperty("model").GetString() ?? "",
                    s.GetProperty("text").GetString() ?? "", s.TryGetProperty("installed", out var i) && i.ValueKind == JsonValueKind.True));
            }
        }

        return found;
    }

    internal static AiResult ParseAiResult(JsonElement r, Func<string, string> relative)
    {
        static IReadOnlyList<string> Strings(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Array
                ? v.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Object ? x.GetProperty("op").GetString() ?? "" : x.GetString() ?? "").ToList() : [];
        var checks = r.TryGetProperty("checks", out var c) && c.ValueKind == JsonValueKind.Object ? c : default;
        FidelityMeasures? fidelity = null;
        if (checks.ValueKind == JsonValueKind.Object && checks.TryGetProperty("detail", out var d) && d.ValueKind == JsonValueKind.Object)
        {
            fidelity = new FidelityMeasures(d.GetProperty("ssim").GetDouble(), 1.0 - Math.Min(1.0, d.GetProperty("lostEdgeRatio").GetDouble()),
                d.GetProperty("newEdgeRatio").GetDouble(), d.GetProperty("meanShift").GetDouble());
        }

        bool? textPassed = checks.ValueKind == JsonValueKind.Object && checks.TryGetProperty("text", out var t)
                           && t.ValueKind == JsonValueKind.Object && t.TryGetProperty("textFound", out var tf) && tf.GetBoolean()
            ? !(t.TryGetProperty("flag", out var flag) && flag.GetBoolean())
            : null;
        string? chemical = checks.ValueKind == JsonValueKind.Object && checks.TryGetProperty("chemical", out var ch) && ch.ValueKind == JsonValueKind.String
            ? ch.GetString() : null;
        string? Rel(string property, string inner) =>
            r.TryGetProperty(property, out var o) && o.ValueKind == JsonValueKind.Object && o.TryGetProperty(inner, out var p) ? relative(p.GetString()!) : null;
        return new AiResult(r.GetProperty("status").GetString() ?? "failed", Strings(r, "models"), Rel("output", "path"),
            Rel("thumbnail", "path"), Strings(r, "reasons"), Strings(r, "operations"), fidelity, textPassed, chemical);
    }

    /// <summary>
    /// Whether AI can be tried on a picture now: a suitable, installed model, not tried yet on this result, and not a
    /// redrawn structure or a sheet of several figures (whose panels are processed one by one).
    /// </summary>
    public static bool IsAiEligible(ImageJob job) =>
        job.Ai is null && job.AiSuggestions.Any(s => s.Installed) && !job.IsRedrawn && !job.Operations.Contains("panels")
        && (job.Status == ProcessingStatus.Completed || job.Status == ProcessingStatus.Skipped && job.Reason == "already_good");

    /// <summary>
    /// Tries AI on the given pictures once processing has finished: each picture shared by several places is processed
    /// once, ineligible pictures are left unchanged. A usable AI version is applied (put in use) when the picture is
    /// still to decide everywhere it appears; the normal result is kept beside it, so the user can switch back. A picture
    /// already approved or kept as original keeps its decision and version until the user chooses. Nothing is approved.
    /// </summary>
    public async Task<AiRunSummary> TryAiAsync(IReadOnlyCollection<string> imageIds, IProgress<(int Done, int Total)>? progress = null,
        CancellationToken ct = default)
    {
        await _reenhanceLock.WaitAsync(ct);
        try
        {
            List<List<ImageJob>> groups;
            int ineligible;
            lock (_stateLock)
            {
                if (State?.Stage != JobStage.AwaitingReview)
                {
                    throw new InvalidOperationException("AI can be tried once processing has finished.");
                }

                var wanted = imageIds.ToHashSet();
                var all = State.Images.Where(i => wanted.Contains(i.Id)).Select(i => i.ResourceRef).Distinct()
                    .Select(r => State.Images.Where(i => i.ResourceRef == r).ToList()).ToList();
                ineligible = all.Count(g => !IsAiEligible(g[0]));
                groups = all.Where(g => IsAiEligible(g[0])).ToList();
            }

            int done = 0, accepted = 0, rejected = 0, applied = 0;
            progress?.Report((0, groups.Count));
            foreach (var group in groups)
            {
                ct.ThrowIfCancellationRequested();
                var result = await RunAiAsync(group[0], ct);
                bool? apply = null;
                Update(group, job =>
                {
                    // Decided once for the whole group, under the state lock: a decision made meanwhile is respected.
                    apply ??= result.Accepted && group.All(j => j.Approval == ApprovalStatus.Undecided);
                    job.Ai = result;
                    job.UseAiVersion = apply.Value;
                    if (result.Accepted && job.OutputPixelSize is null)
                    {
                        job.OutputPixelSize = job.PixelSize;
                    }
                });
                applied += apply == true ? 1 : 0;
                accepted += result.Accepted ? 1 : 0;
                rejected += result.Status == "rejected" ? 1 : 0;
                progress?.Report((++done, groups.Count));
            }

            await SaveAsync(ct);
            return new AiRunSummary(groups.Count, accepted, rejected, ineligible, applied);
        }
        finally
        {
            _reenhanceLock.Release();
        }
    }

    private async Task<AiResult> RunAiAsync(ImageJob lead, CancellationToken ct)
    {
        var output = Workspace!.Resolve(WorkspaceFolder.Enhanced, lead.ResourceId + "_ai" + ExtensionFor(lead.OutputFormat));
        var thumbnail = Workspace.Resolve(WorkspaceFolder.Thumbnails, lead.ResourceId + "_ai.jpg");
        try
        {
            var result = await (await WorkerAsync(0, ct)).SendAsync("ai_enhance", new
            {
                input = Input(lead),
                output,
                format = lead.OutputFormat,
                frame = lead.Location.FrameIndex ?? 0,
                settings = lead.Settings,
                classification = ClassificationRequest(lead),
                opencv = lead.Status == ProcessingStatus.Completed && lead.EnhancedPath is not null ? Abs(lead.EnhancedPath) : null,
                thumbnails = new { ai = thumbnail, maxSide = _options.ThumbnailSize },
            }, ct: ct);
            return ParseAiResult(result, Workspace.ToRelative);
        }
        catch (WorkerException ex)
        {
            return new AiResult("failed", [], null, null, [ex.Message], [], null, null, null);
        }
        catch (WorkerUnavailableException) when (!ct.IsCancellationRequested)
        {
            await RestartWorkerAsync(0);
            return new AiResult("failed", [], null, null, ["the image processor stopped while trying AI on this picture"], [], null, null, null);
        }
    }

    /// <summary>
    /// The version the user keeps for a picture (and every place it appears): the AI version (only an accepted one) or
    /// the normal result. Changing the version of an approved picture changes what was approved, so it is only done
    /// when the user confirmed it (<paramref name="approvalChangeConfirmed"/>); the picture then needs approving again.
    /// Returns false, changing nothing, when that confirmation is still needed.
    /// </summary>
    public async Task<bool> ChooseVersionAsync(string imageId, bool useAi, bool approvalChangeConfirmed = false, CancellationToken ct = default)
    {
        List<ImageJob> group;
        lock (_stateLock)
        {
            var image = State!.Images.Single(i => i.Id == imageId);
            if (useAi && image.Ai is not { Accepted: true })
            {
                throw new InvalidOperationException("There is no usable AI version of this picture.");
            }

            group = State.Images.Where(i => i.ResourceRef == image.ResourceRef).ToList();
            if (group.All(j => j.UseAiVersion == useAi))
            {
                return true;
            }

            if (group.Any(j => j.Approval == ApprovalStatus.Approved) && !approvalChangeConfirmed)
            {
                return false;
            }
        }

        Update(group, job =>
        {
            job.UseAiVersion = useAi;
            if (job.Approval == ApprovalStatus.Approved)
            {
                job.Approval = ApprovalStatus.Undecided; // confirmed by the user: the new version needs its own approval
            }
        });
        await SaveAsync(ct);
        return true;
    }

    // ------------------------------------------------------------------ enhance

    private async Task EnhanceAsync(CancellationToken ct)
    {
        var slots = new ConcurrentQueue<int>(Enumerable.Range(0, Math.Max(1, _options.WorkerCount)));
        while (true)
        {
            await _gate.WaitIfPausedAsync(ct);
            ct.ThrowIfCancellationRequested();

            if (_workspaces.GetAvailableFreeSpace() < _options.MinFreeDiskBytes)
            {
                _gate.Pause("Low disk space: free some space on the drive, then resume.");
                Report(force: true);
                continue;
            }

            List<List<ImageJob>> pending;
            lock (_stateLock)
            {
                pending = State!.Images
                    .Where(i => i.Status == ProcessingStatus.Pending)
                    .GroupBy(i => i.ResourceRef)
                    .Select(g => g.ToList())
                    .ToList();
            }

            if (pending.Count == 0)
            {
                break;
            }

            var available = await AvailableMemoryAsync(ct);
            var budget = BatchPlanner.Budget(available, _options.AvailableRamFraction, _options.MaxImageMemoryBytes, _options.WorkerCount);
            var perImage = BatchPlanner.PerImageLimit(budget, _options.MaxImageMemoryBytes, _options.WorkerCount);
            var batch = BatchPlanner.NextBatch(pending, g => g[0].EstimatedMemoryBytes ?? 0, budget, _options.MaxBatchSize);

            await Parallel.ForEachAsync(batch,
                new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, _options.WorkerCount), CancellationToken = ct },
                async (group, token) =>
                {
                    await _gate.WaitIfPausedAsync(token);
                    slots.TryDequeue(out var slot);
                    try
                    {
                        await EnhanceGroupAsync(group, slot, perImage, token);
                    }
                    finally
                    {
                        slots.Enqueue(slot);
                    }
                });

            lock (_stateLock)
            {
                State!.Performance.BatchCount++;
            }

            await SaveAsync(ct);
        }
    }

    private async Task EnhanceGroupAsync(List<ImageJob> group, int slot, long perImageLimit, CancellationToken ct)
    {
        var lead = group[0];
        Update(group, job =>
        {
            job.Status = ProcessingStatus.Processing;
            job.Attempts++;
        });
        var number = Count(s => s is ProcessingStatus.Completed or ProcessingStatus.Skipped or ProcessingStatus.Failed) + 1;
        SetStage(JobStage.Enhancing, number - 1, State!.Images.Count, $"Enhancing image {number}");
        Report();

        var input = lead.ExtractedPath is not null ? Abs(lead.ExtractedPath) : lead.Location.SourcePath!;
        var extension = ExtensionFor(lead.OutputFormat);
        var output = Workspace!.Resolve(WorkspaceFolder.Enhanced, lead.ResourceId + extension);
        var originalThumb = Workspace.Resolve(WorkspaceFolder.Thumbnails, lead.ResourceId + "_original.jpg");
        var enhancedThumb = Workspace.Resolve(WorkspaceFolder.Thumbnails, lead.ResourceId + "_enhanced.jpg");
        var sw = Stopwatch.StartNew();
        JsonElement result;
        try
        {
            var worker = await WorkerAsync(slot, ct);
            result = await worker.SendAsync("enhance", new
            {
                input,
                output,
                format = lead.OutputFormat,
                frame = lead.Location.FrameIndex ?? 0,
                // Widest place the image is shown, so low-resolution images get fine-detail processing
                // (the output keeps the original pixel size).
                displayWidthPt = group.Max(j => j.Location.Placements?.Max(p => p.Rect[2] - p.Rect[0]) ?? j.DisplaySize?.Width),
                // Image files have no displayed size: small ones get fine-detail processing.
                standalone = lead.SourceType == SourceType.Image,
                settings = lead.Settings,
                // Redraw (and so MolScribe) is only tried where this says "chemical".
                classification = ClassificationRequest(lead),
                limits = new { maxPixels = _options.MaxPixels, maxMemoryBytes = perImageLimit },
                thumbnails = new { original = originalThumb, enhanced = enhancedThumb, maxSide = _options.ThumbnailSize },
            }, ct: ct);
        }
        catch (WorkerException ex)
        {
            Fail(group, ex.Code, ex.Message);
            return;
        }
        catch (WorkerUnavailableException) when (!ct.IsCancellationRequested && !_cancelRequested)
        {
            // The worker died on this image (e.g. a decoder crash). Restart it; retry once, then give up on this image.
            await RestartWorkerAsync(slot);
            if (lead.Attempts < 2)
            {
                Update(group, job => job.Status = ProcessingStatus.Pending);
            }
            else
            {
                Fail(group, "worker_crashed", "the image processor stopped unexpectedly while working on this image");
            }

            return;
        }

        sw.Stop();
        _eta.Add(sw.Elapsed);
        ApplyResult(group, result, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>The "findings" of a redraw outcome: plain sentences, each with its place in the picture when known.</summary>
    private static List<RedrawFinding> Findings(JsonElement redraw)
    {
        var found = new List<RedrawFinding>();
        if (redraw.ValueKind != JsonValueKind.Object || !redraw.TryGetProperty("findings", out var list) || list.ValueKind != JsonValueKind.Array)
        {
            return found;
        }

        foreach (var item in list.EnumerateArray())
        {
            var message = item.TryGetProperty("message", out var m) ? m.GetString() ?? "" : "";
            double[]? box = item.TryGetProperty("box", out var b) && b.ValueKind == JsonValueKind.Array && b.GetArrayLength() == 4
                ? b.EnumerateArray().Select(v => v.GetDouble()).ToArray()
                : null;
            if (message.Length > 0)
            {
                found.Add(new RedrawFinding(message, box));
            }
        }

        return found;
    }

    private static bool? IsChemical(JsonElement redraw) =>
        redraw.TryGetProperty("isChemical", out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean() : null;

    private void ApplyResult(List<ImageJob> group, JsonElement result, double milliseconds)
    {
        var ws = Workspace!;
        var status = result.GetProperty("status").GetString();
        string? Rel(string property) =>
            result.TryGetProperty(property, out var t) && t.TryGetProperty("path", out var p) ? ws.ToRelative(p.GetString()!) : null;
        var originalThumb = Rel("originalThumbnail");
        var enhancedThumb = Rel("enhancedThumbnail");
        var content = result.TryGetProperty("analysis", out var analysis) ? analysis.GetProperty("content").GetString() : null;
        static QualityMeasures? Quality(JsonElement q) =>
            q.ValueKind != JsonValueKind.Object || !q.TryGetProperty("sharpness", out _)
                ? null
                : new QualityMeasures(q.GetProperty("sharpness").GetDouble(), q.GetProperty("noise").GetDouble(),
                    q.TryGetProperty("blur", out var b) ? b.GetDouble() : 0, q.GetProperty("dynamicRange").GetDouble());
        var before = analysis.ValueKind == JsonValueKind.Object ? Quality(analysis) : null;
        var after = result.TryGetProperty("after", out var afterElement) ? Quality(afterElement) : null;
        TextCheckResult? textCheck = null;
        if (result.TryGetProperty("textCheck", out var tc) && tc.ValueKind == JsonValueKind.Object
            && tc.TryGetProperty("textFound", out var found))
        {
            static IReadOnlyList<string> Strings(JsonElement e, string name) =>
                e.TryGetProperty(name, out var list) ? list.EnumerateArray().Select(x => x.GetString() ?? "").ToList() : [];
            static double? Number(JsonElement e, string name) =>
                e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
            textCheck = new TextCheckResult(found.GetBoolean(), tc.TryGetProperty("flag", out var f) && f.GetBoolean(),
                tc.TryGetProperty("characters", out var c) ? c.GetInt32() : 0,
                tc.TryGetProperty("changedCharacters", out var ch) ? ch.GetInt32() : 0,
                Strings(tc, "numbersChanged"), Strings(tc, "examples"), Number(tc, "confidenceBefore"), Number(tc, "confidenceAfter"),
                Strings(tc, "numbersUncertain"));
        }

        RedrawResult? redraw = null;
        if (result.TryGetProperty("redraw", out var rd) && rd.ValueKind == JsonValueKind.Object && rd.TryGetProperty("redrawn", out var redrawn))
        {
            static IReadOnlyList<string> List(JsonElement e, string name) =>
                e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var list) && list.ValueKind == JsonValueKind.Array
                    ? list.EnumerateArray().Select(x => x.GetString() ?? "").ToList() : [];
            static string? Text(JsonElement e, string name) =>
                e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
            static double? Value(JsonElement e, string name) =>
                e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
            var evidence = rd.TryGetProperty("evidence", out var ev) ? ev : default;
            var molecule = evidence.ValueKind == JsonValueKind.Object && evidence.TryGetProperty("molecule", out var mo) ? mo : default;
            redraw = new RedrawResult(redrawn.GetBoolean(), List(rd, "reasons"), Text(molecule, "smiles"), Text(molecule, "inchiKey"),
                Text(molecule, "formula"), Value(molecule, "molecularWeight"), List(molecule, "stereocentres"),
                Value(evidence, "confidence"), Text(evidence, "note"),
                !(rd.TryGetProperty("proven", out var proven) && proven.ValueKind == JsonValueKind.False),
                List(rd, "unproven"), Text(evidence, "layout"), Findings(rd), IsChemical(rd));
        }

        var panels = new List<PanelResult>();
        if (result.TryGetProperty("panels", out var ps) && ps.ValueKind == JsonValueKind.Array)
        {
            foreach (var panel in ps.EnumerateArray())
            {
                RedrawResult? panelRedraw = null;
                if (panel.TryGetProperty("redraw", out var pr) && pr.ValueKind == JsonValueKind.Object && pr.TryGetProperty("redrawn", out var prd))
                {
                    var pev = pr.TryGetProperty("evidence", out var e) ? e : default;
                    var pmol = pev.ValueKind == JsonValueKind.Object && pev.TryGetProperty("molecule", out var pm) ? pm : default;
                    string? S(JsonElement el, string name) =>
                        el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
                    double? N(JsonElement el, string name) =>
                        el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
                    IReadOnlyList<string> L(JsonElement el, string name) =>
                        el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Array
                            ? v.EnumerateArray().Select(x => x.GetString() ?? "").ToList() : [];
                    panelRedraw = new RedrawResult(prd.GetBoolean(), L(pr, "reasons"), S(pmol, "smiles"), S(pmol, "inchiKey"), S(pmol, "formula"),
                        N(pmol, "molecularWeight"), L(pmol, "stereocentres"), N(pev, "confidence"), S(pev, "note"),
                        !(pr.TryGetProperty("proven", out var pp) && pp.ValueKind == JsonValueKind.False),
                        L(pr, "unproven"), S(pev, "layout"), Findings(pr), IsChemical(pr));
                }

                panels.Add(new PanelResult(panel.GetProperty("panel").GetInt32(), panel.GetProperty("content").GetString() ?? "",
                    panel.GetProperty("operations").EnumerateArray().Select(o => o.GetString() ?? "").ToList(), panelRedraw));
            }
        }

        VersionChoice? choice = null;
        if (result.TryGetProperty("choice", out var cho) && cho.ValueKind == JsonValueKind.Object
            && cho.TryGetProperty("chosen", out var chosen) && cho.TryGetProperty("candidates", out var cands))
        {
            choice = new VersionChoice(chosen.GetString() ?? "",
                cands.EnumerateArray().Select(c => new VersionCandidate(
                    c.GetProperty("name").GetString() ?? "",
                    c.TryGetProperty("noise", out var n) && n.ValueKind == JsonValueKind.Number ? n.GetDouble() : 0,
                    c.TryGetProperty("legibility", out var l) && l.ValueKind == JsonValueKind.Number ? l.GetDouble() : null,
                    c.TryGetProperty("rejected", out var r) && r.ValueKind == JsonValueKind.String ? r.GetString() : null)).ToList(),
                cho.TryGetProperty("lessReadable", out var lr) && lr.ValueKind == JsonValueKind.True);
        }

        var aiSuggestions = ParseSuggestions(result);
        var aiDevices = new Dictionary<string, string>();
        if (result.TryGetProperty("aiDevices", out var devices) && devices.ValueKind == JsonValueKind.Object)
        {
            foreach (var d in devices.EnumerateObject())
            {
                aiDevices[d.Name] = d.Value.GetString() ?? "";
            }
        }

        FidelityMeasures? fidelity = null;
        if (result.TryGetProperty("metrics", out var m))
        {
            fidelity = new FidelityMeasures(m.GetProperty("ssim").GetDouble(), 1.0 - Math.Min(1.0, m.GetProperty("lostEdgeRatio").GetDouble()),
                m.GetProperty("newEdgeRatio").GetDouble(), m.GetProperty("meanShift").GetDouble());
        }

        lock (_stateLock)
        {
            var perf = State!.Performance;
            perf.TotalEnhanceMilliseconds += milliseconds;
            if (result.TryGetProperty("memory", out var memory))
            {
                perf.PeakWorkerMemoryBytes = Math.Max(perf.PeakWorkerMemoryBytes, memory.GetProperty("peakRssBytes").GetInt64());
            }

            if (status == "completed")
            {
                perf.ImagesEnhanced++;
            }

            foreach (var job in group)
            {
                job.OriginalThumbnailPath = originalThumb;
                job.ContentKind = content;
                job.QualityBefore = before;
                job.QualityAfter = status == "completed" ? after : null;
                job.Fidelity = status == "completed" ? fidelity : null;
                job.TextCheck = status == "completed" ? textCheck : null;
                job.Redraw = redraw;
                job.Panels = panels;
                job.Choice = status == "completed" ? choice : null;
                job.AiDevices = status == "completed" ? new Dictionary<string, string>(aiDevices) : [];
                job.AiSuggestions = [.. aiSuggestions];
                job.EnhanceMilliseconds = milliseconds;
                if (status == "completed")
                {
                    var output = result.GetProperty("output");
                    job.Status = ProcessingStatus.Completed;
                    job.EnhancedPath = ws.ToRelative(output.GetProperty("path").GetString()!);
                    job.OutputPixelSize = new PixelSize(output.GetProperty("width").GetInt32(), output.GetProperty("height").GetInt32());
                    job.EnhancedThumbnailPath = enhancedThumb;
                    job.NeedsReview = result.GetProperty("needsReview").GetBoolean();
                    job.ReviewReasons = result.GetProperty("reviewReasons").EnumerateArray().Select(r => r.GetString()!).ToList();
                    job.Operations = result.GetProperty("operations").EnumerateArray().Select(o => o.GetProperty("op").GetString()!).ToList();
                    job.Notes = result.GetProperty("notes").EnumerateArray().Select(n => n.GetString()!).ToList();
                }
                else
                {
                    job.Status = ProcessingStatus.Skipped;
                    job.Reason = result.GetProperty("reason").GetString();
                    job.Error = result.GetProperty("message").GetString();
                }
            }
        }

        foreach (var job in group)
        {
            ImageUpdated?.Invoke(job.Clone());
        }

        Report();
    }

    // ------------------------------------------------------------------ workers

    private async Task<PythonWorkerClient> WorkerAsync(int slot, CancellationToken ct)
    {
        lock (_stateLock)
        {
            if (_workers.Length == 0)
            {
                _workers = new PythonWorkerClient?[Math.Max(1, _options.WorkerCount)];
            }
        }

        var worker = _workers[slot];
        if (worker is { IsRunning: true })
        {
            return worker;
        }

        if (worker is not null)
        {
            await worker.DisposeAsync();
        }

        worker = _workerFactory();
        _workers[slot] = worker;
        await worker.StartAsync(ct);
        if (_options.WorkerCount > 1)
        {
            // Parallel workers each get a share of the cores instead of OpenCV's default of all cores.
            await worker.SendAsync("configure", new { cvThreads = Math.Max(1, Environment.ProcessorCount / _options.WorkerCount) }, ct: ct);
        }

        return worker;
    }

    private async Task RestartWorkerAsync(int slot)
    {
        var worker = _workers[slot];
        _workers[slot] = null;
        if (worker is not null)
        {
            await worker.DisposeAsync();
        }

        lock (_stateLock)
        {
            State!.Performance.WorkerRestarts++;
        }
    }

    private async Task StopWorkersAsync(TimeSpan grace)
    {
        var workers = _workers.Where(w => w is not null).Cast<PythonWorkerClient>().ToList();
        _workers = workers.Count == 0 ? _workers : new PythonWorkerClient?[_workers.Length];
        await Task.WhenAll(workers.Select(async w =>
        {
            var stop = w.StopAsync();
            if (await Task.WhenAny(stop, Task.Delay(grace)) != stop)
            {
                w.Kill();
            }

            await w.DisposeAsync();
        }));
    }

    private async Task<long?> AvailableMemoryAsync(CancellationToken ct)
    {
        try
        {
            var memory = await (await WorkerAsync(0, ct)).SendAsync("memory", ct: ct);
            return memory.GetProperty("availableBytes").ValueKind == JsonValueKind.Number
                ? memory.GetProperty("availableBytes").GetInt64()
                : null;
        }
        catch (WorkerException)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------ state helpers

    private void Update(IEnumerable<ImageJob> jobs, Action<ImageJob> change)
    {
        List<ImageJob> snapshots;
        lock (_stateLock)
        {
            foreach (var job in jobs)
            {
                change(job);
            }

            snapshots = jobs.Select(j => j.Clone()).ToList();
        }

        foreach (var snapshot in snapshots)
        {
            ImageUpdated?.Invoke(snapshot);
        }
    }

    private void Fail(IEnumerable<ImageJob> jobs, string code, string message) =>
        Update(jobs, job =>
        {
            job.Status = ProcessingStatus.Failed;
            job.Reason = code;
            job.Error = message;
        });

    private static void Skip(ImageJob job, string reason, string message)
    {
        job.Status = ProcessingStatus.Skipped;
        job.Reason = reason;
        job.Error = message;
    }

    private async Task SaveAsync(CancellationToken ct)
    {
        if (State is null || Workspace is null || !Directory.Exists(Workspace.RootPath))
        {
            return;
        }

        byte[] json;
        lock (_stateLock)
        {
            json = JobStateStore.Serialize(State);
        }

        await _saveLock.WaitAsync(ct);
        try
        {
            await JobStateStore.SaveAsync(Workspace, json, ct);
        }
        finally
        {
            _saveLock.Release();
        }
    }

    private void SetStage(JobStage stage, int done, int total, string operation)
    {
        _stage = (stage, done, total, operation);
    }

    private int Count(Func<ProcessingStatus, bool> predicate)
    {
        lock (_stateLock)
        {
            return State?.Images.Count(i => predicate(i.Status)) ?? 0;
        }
    }

    private void Report(bool force = false)
    {
        var handler = ProgressChanged;
        if (handler is null || State is null)
        {
            return;
        }

        lock (_sinceReport)
        {
            if (!force && _sinceReport.Elapsed.TotalMilliseconds < ReportInterval)
            {
                return;
            }

            _sinceReport.Restart();
        }

        JobProgress progress;
        lock (_stateLock)
        {
            var images = State.Images;
            int completed = 0, skipped = 0, failed = 0, pending = 0;
            foreach (var image in images)
            {
                switch (image.Status)
                {
                    case ProcessingStatus.Completed: completed++; break;
                    case ProcessingStatus.Skipped: skipped++; break;
                    case ProcessingStatus.Failed: failed++; break;
                    case ProcessingStatus.Pending or ProcessingStatus.Processing: pending++; break;
                }
            }

            var processed = completed + skipped + failed;
            var (stage, done, total, operation) = _stage;
            var stageFraction = total > 0 ? (double)done / total : 0;
            // Stage weights: copy 0-2 %, scan 2-10 %, extract 10-20 %, classify 20-25 %, enhance 25-99 %.
            // 100 % only when finished.
            var percent = State.Stage switch
            {
                JobStage.Created => 2 * stageFraction,
                JobStage.Scanning => 2 + 8 * stageFraction,
                JobStage.Extracting => 10 + 10 * stageFraction,
                JobStage.Classifying => 20 + 5 * stageFraction,
                JobStage.Enhancing => 25 + 74 * (images.Count == 0 ? 1 : (double)processed / images.Count),
                JobStage.AwaitingReview => 100,
                // A separate run: 100 % only after the output has been validated and saved.
                JobStage.Exporting => Math.Min(99, 99 * stageFraction),
                JobStage.Completed => 100,
                _ => 0,
            };
            progress = new JobProgress
            {
                DocumentName = State.SourceFileName,
                Stage = State.Stage,
                StageDone = done,
                StageTotal = total,
                TotalImages = images.Count,
                Processed = processed,
                Pending = pending,
                Completed = completed,
                Skipped = skipped,
                Failed = failed,
                Percent = Math.Round(percent, 1),
                CurrentOperation = _gate.IsPaused ? "Paused" : operation,
                EstimatedRemaining = State.Stage == JobStage.Enhancing
                    ? _eta.Estimate(images.Where(i => i.Status is ProcessingStatus.Pending or ProcessingStatus.Processing)
                        .Select(i => i.ResourceRef).Distinct().Count(), _options.WorkerCount)
                    : null,
                IsPaused = _gate.IsPaused,
                PauseReason = _gate.Reason,
            };
        }

        handler(progress);
    }

    private string Abs(string relative) => Path.Combine(Workspace!.RootPath, relative);

    private static string NextId(List<ImageJob> jobs) => $"img_{jobs.Count + 1:D5}";

    private static string ResourceIdFor(Dictionary<string, string> map, string key) =>
        map.TryGetValue(key, out var id) ? id : map[key] = $"r{map.Count + 1:D5}";

    private static string SafeExtension(string uri)
    {
        var ext = Path.GetExtension(uri).ToLowerInvariant();
        return ext.Length is > 1 and <= 5 && ext[1..].All(char.IsAsciiLetterOrDigit) ? ext : ".img";
    }

    private static string ExtensionFor(string? format) => format switch
    {
        "jpeg" => ".jpg",
        "png" => ".png",
        "bmp" => ".bmp",
        "tiff" => ".tif",
        "webp" => ".webp",
        "jp2" => ".jp2",
        "pnm" => ".ppm",
        "gif" => ".png",
        _ => ".png",
    };

    private static string DocxSkipMessage(string reason) => reason switch
    {
        "ole_preview" => "preview of an embedded object (e.g. a ChemDraw structure); its program redraws it",
        "svg_fallback" => "backup copy of an SVG picture; Word shows the SVG",
        "linked_image" => "picture is linked from outside the document",
        "vector" => "vector graphics are kept unchanged",
        "missing_image" => "the picture is missing from the document",
        _ => "this image format is kept unchanged",
    };

    // ------------------------------------------------------------------ input detection and errors

    internal static SourceType DetectSourceType(IReadOnlyList<string> files)
    {
        if (files.Count == 0)
        {
            throw new JobFailure("no_input", "No file was selected.");
        }

        foreach (var file in files)
        {
            if (!File.Exists(file))
            {
                throw new JobFailure("file_not_found", $"{Path.GetFileName(file)} does not exist.");
            }
        }

        var kinds = files.Select(Classify).ToList();
        if (kinds.All(k => k == SourceType.Image))
        {
            return SourceType.Image;
        }

        if (files.Count == 1)
        {
            return kinds[0];
        }

        throw new JobFailure("unsupported_input", "Choose one PDF or Word document, or any number of image files.");

        static SourceType Classify(string file)
        {
            Span<byte> head = stackalloc byte[8];
            int read;
            using (var stream = File.OpenRead(file))
            {
                read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
            }

            head = head[..read];
            if (head.StartsWith("%PDF"u8))
            {
                return SourceType.Pdf;
            }

            var ext = Path.GetExtension(file).ToLowerInvariant();
            if (ext is ".docx" or ".docm" or ".doc" || head.StartsWith(new byte[] { 0xD0, 0xCF, 0x11, 0xE0 }))
            {
                return SourceType.Docx; // the DOCX scanner explains .doc and encrypted files
            }

            if (head.StartsWith("PK"u8))
            {
                using var zip = ZipFile.OpenRead(file);
                if (zip.GetEntry("word/document.xml") is not null)
                {
                    return SourceType.Docx;
                }
            }

            return SourceType.Image;
        }
    }

    private static JobFailure? ToFailure(Exception ex) => ex switch
    {
        JobFailure f => f,
        WorkerException w => new JobFailure(w.Code, w.Message, w),
        DocxException d => new JobFailure(d.Code, d.Message, d),
        WorkerUnavailableException u => new JobFailure("worker_unavailable", "The image processor could not run: " + u.Message, u),
        IOException io when (io.HResult & 0xFFFF) is 0x70 or 0x27 => new JobFailure("disk_full", "The disk is full.", io),
        IOException io => new JobFailure("io_error", io.Message, io),
        UnauthorizedAccessException ua => new JobFailure("access_denied", ua.Message, ua),
        _ => null,
    };
}
