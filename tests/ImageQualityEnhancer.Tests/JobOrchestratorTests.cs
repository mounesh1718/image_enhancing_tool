using System.Collections.Concurrent;
using System.Security.Cryptography;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>Input files generated once for all orchestrator tests.</summary>
public sealed class OrchestratorFixture : IDisposable
{
    private readonly TempDirectory _dir = new();

    public OrchestratorFixture()
    {
        Images = Path.Combine(_dir.Path, "images");
        Directory.CreateDirectory(Images);
        TestSamples.ImageSet(Images);
        Pdf = Path.Combine(_dir.Path, "report.pdf");
        TestSamples.FeaturePdf(Pdf);
        SecretPdf = Path.Combine(_dir.Path, "secret.pdf");
        TestSamples.FeaturePdf(SecretPdf, "s3cret");
        Docx = Path.Combine(_dir.Path, "report.docx");
        DocxSamples.Build(Docx, Images);
    }

    public string Images { get; }
    public string Pdf { get; }
    public string SecretPdf { get; }
    public string Docx { get; }

    public void Dispose() => _dir.Dispose();
}

public class JobOrchestratorTests : IClassFixture<OrchestratorFixture>, IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromMinutes(3);
    private readonly OrchestratorFixture _fx;
    private readonly TempDirectory _root = new();
    private readonly WorkspaceManager _workspaces;

    public JobOrchestratorTests(OrchestratorFixture fixture)
    {
        _fx = fixture;
        _workspaces = new WorkspaceManager(Path.Combine(_root.Path, "jobs"));
    }

    public void Dispose() => _root.Dispose();

    private string Input(string name) => Path.Combine(_root.Path, name);

    /// <summary>Copies of the noisy test photo under distinct names, so each is a separate image to enhance.</summary>
    private List<string> Photos(int count, string prefix = "photo")
    {
        Directory.CreateDirectory(Input("in"));
        return Enumerable.Range(1, count).Select(i =>
        {
            var path = Path.Combine(Input("in"), $"{prefix}{i}.png");
            File.Copy(Path.Combine(_fx.Images, "photo.png"), path, overwrite: true);
            return path;
        }).ToList();
    }

    private static byte[] Sha(string path) => SHA256.HashData(File.ReadAllBytes(path));

    // ---------------------------------------------------------------- full runs

    [Fact]
    public async Task ImageFiles_RunToReview_WithPerImageOutcomes()
    {
        Directory.CreateDirectory(Input("mixed"));
        var files = new List<string>();
        foreach (var name in new[] { "photo.png", "photo.jpg", "scan.png", "small.gif" })
        {
            files.Add(Path.Combine(Input("mixed"), name));
            File.Copy(Path.Combine(_fx.Images, name), files[^1]);
        }

        files.Add(Input("mixed/broken.jpg"));
        File.WriteAllBytes(files[^1], [0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 1, 2, 3]);
        files.Add(Input("mixed/diagram.svg"));
        File.WriteAllText(files[^1], "<svg xmlns=\"http://www.w3.org/2000/svg\"/>");
        var hashes = files.ToDictionary(f => f, Sha);

        var progress = new ConcurrentQueue<JobProgress>();
        await using var job = new JobOrchestrator(_workspaces);
        job.ProgressChanged += progress.Enqueue;
        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        var byName = job.State!.Images.ToDictionary(i => Path.GetFileName(i.Location.SourcePath!));
        Assert.Equal(ProcessingStatus.Completed, byName["photo.png"].Status);
        Assert.Equal(ProcessingStatus.Completed, byName["scan.png"].Status);
        Assert.True(byName["scan.png"].NeedsReview); // diagram content always needs review
        Assert.Equal(ProcessingStatus.Failed, byName["broken.jpg"].Status);
        Assert.Equal("decode_failed", byName["broken.jpg"].Reason);
        Assert.Equal((ProcessingStatus.Skipped, "vector"), (byName["diagram.svg"].Status, byName["diagram.svg"].Reason));
        Assert.EndsWith(".png", byName["small.gif"].EnhancedPath); // enhanced GIF stored as PNG

        foreach (var image in job.State.Images.Where(i => i.Status == ProcessingStatus.Completed))
        {
            Assert.True(File.Exists(Path.Combine(job.Workspace!.RootPath, image.EnhancedPath!)));
            Assert.True(File.Exists(Path.Combine(job.Workspace.RootPath, image.EnhancedThumbnailPath!)));
        }

        Assert.All(files, f => Assert.Equal(hashes[f], Sha(f))); // inputs never modified

        // Progress never goes backwards and only reaches 100 % when processing is finished.
        var events = progress.ToList();
        Assert.True(events.Zip(events.Skip(1)).All(p => p.Second.Percent >= p.First.Percent));
        Assert.All(events.Where(e => e.Stage != JobStage.AwaitingReview), e => Assert.True(e.Percent < 100));
        Assert.Equal(100, events[^1].Percent);

        var saved = await JobStateStore.LoadAsync(job.Workspace!);
        Assert.Equal(JobStage.AwaitingReview, saved!.Stage);
        Assert.Equal(job.State.Images.Count, saved.Images.Count);
        Assert.True(saved.Performance.ImagesEnhanced >= 3);
    }

    [Fact]
    public async Task Pdf_RunToReview_EnhancesSharedImagesOnce()
    {
        await using var job = new JobOrchestrator(_workspaces);
        var outcome = await job.RunAsync(new JobRequest([_fx.Pdf])).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        var state = job.State!;
        Assert.Equal(Convert.ToHexString(Sha(_fx.Pdf)), state.SourceSha256);
        var reused = state.Images.Single(i => i.Location.Placements!.Count == 2); // the photo shown on two pages
        Assert.Equal(ProcessingStatus.Completed, reused.Status);

        // Identical pixels stored as two objects: one enhancement, shared result.
        var twins = state.Images.Where(i => i.ResourceRef == reused.ResourceRef).ToList();
        Assert.Equal(2, twins.Count);
        Assert.Single(twins.Select(t => t.EnhancedPath).Distinct());

        Assert.Contains(state.Images, i => i.Reason == "unsupported_colorspace");
        Assert.Contains(state.Images, i => i.Reason == "inline_image");
        Assert.Equal(state.Images.Where(i => i.Status == ProcessingStatus.Completed).Select(i => i.ResourceRef).Distinct().Count(),
            state.Performance.ImagesEnhanced);
    }

    [Fact]
    public async Task Docx_RunToReview_OneJobPerOccurrence()
    {
        await using var job = new JobOrchestrator(_workspaces);
        var outcome = await job.RunAsync(new JobRequest([_fx.Docx])).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        var images = job.State!.Images;
        var photo = images.Where(i => i.Location.ImageUri == images.Single(x => x.Location.InTable).Location.ImageUri).ToList();
        Assert.Equal(4, photo.Count); // table, floating, compatibility block, header
        Assert.Single(photo.Select(p => p.EnhancedPath).Distinct());
        Assert.Contains(images, i => i.Reason == "ole_preview");
        Assert.Contains(images, i => i.Reason == "svg_fallback");
        Assert.Contains(images, i => i.Reason == "linked_image");
    }

    // ---------------------------------------------------------------- control

    [Fact]
    public async Task PauseStopsNewImages_ResumeFinishes()
    {
        var files = Photos(6);
        await using var job = new JobOrchestrator(_workspaces, new ProcessingOptions { MaxBatchSize = 2 });
        var firstDone = new TaskCompletionSource();
        job.ImageUpdated += image =>
        {
            if (image.Status == ProcessingStatus.Completed && firstDone.TrySetResult())
            {
                job.Pause(); // once, on the first finished image
            }
        };

        var run = job.RunAsync(new JobRequest(files));
        await firstDone.Task.WaitAsync(Timeout);
        // Images already running when Pause was pressed (one per worker) are allowed to finish.
        var deadline = DateTime.UtcNow + Timeout;
        while (job.State!.Images.Any(i => i.Status == ProcessingStatus.Processing) && DateTime.UtcNow < deadline)
        {
            await Task.Delay(50);
        }

        var completedWhilePaused = job.State!.Images.Count(i => i.Status == ProcessingStatus.Completed);
        await Task.Delay(1500);

        Assert.True(job.IsPaused);
        Assert.False(run.IsCompleted);
        Assert.Equal(completedWhilePaused, job.State.Images.Count(i => i.Status == ProcessingStatus.Completed));
        Assert.True(completedWhilePaused < 6);

        job.Resume();
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await run.WaitAsync(Timeout)).Kind);
        Assert.All(job.State.Images, i => Assert.Equal(ProcessingStatus.Completed, i.Status));
    }

    [Fact]
    public async Task Cancel_StopsJob_AndRemovesWorkspace()
    {
        var files = Photos(8);
        await using var job = new JobOrchestrator(_workspaces);
        var started = new TaskCompletionSource();
        job.ImageUpdated += image =>
        {
            if (image.Status == ProcessingStatus.Completed)
            {
                started.TrySetResult();
            }
        };

        var run = job.RunAsync(new JobRequest(files));
        await started.Task.WaitAsync(Timeout);
        await job.CancelAsync();

        Assert.Equal(JobOutcomeKind.Cancelled, (await run.WaitAsync(Timeout)).Kind);
        Assert.Equal(JobStage.Cancelled, job.State!.Stage);
        Assert.Contains(job.State.Images, i => i.Status == ProcessingStatus.Cancelled);
        Assert.False(Directory.Exists(job.Workspace!.RootPath));
        Assert.All(files, f => Assert.True(File.Exists(f)));
    }

    [Fact]
    public async Task Interrupted_JobResumesWithoutRedoingFinishedImages()
    {
        var files = Photos(5);
        using var appClosing = new CancellationTokenSource();
        JobWorkspace workspace;
        Dictionary<string, double?> finishedBefore;
        await using (var first = new JobOrchestrator(_workspaces, new ProcessingOptions { MaxBatchSize = 1 }))
        {
            first.ImageUpdated += image =>
            {
                if (image.Status == ProcessingStatus.Completed)
                {
                    appClosing.Cancel();
                }
            };
            var outcome = await first.RunAsync(new JobRequest(files), appClosing.Token).WaitAsync(Timeout);
            Assert.Equal(JobOutcomeKind.Interrupted, outcome.Kind);
            workspace = first.Workspace!;
            finishedBefore = first.State!.Images.Where(i => i.Status == ProcessingStatus.Completed)
                .ToDictionary(i => i.Id, i => i.EnhanceMilliseconds);
            Assert.NotEmpty(finishedBefore);
            Assert.Contains(first.State.Images, i => i.Status == ProcessingStatus.Pending);
        }

        Assert.NotNull(_workspaces.Open(workspace.JobId)); // kept for resume
        await using var second = new JobOrchestrator(_workspaces);
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await second.ResumeAsync(workspace).WaitAsync(Timeout)).Kind);
        Assert.All(second.State!.Images, i => Assert.Equal(ProcessingStatus.Completed, i.Status));
        foreach (var (id, ms) in finishedBefore)
        {
            var image = second.State.Images.Single(i => i.Id == id);
            Assert.Equal(ms, image.EnhanceMilliseconds);
            Assert.Equal(1, image.Attempts);
        }
    }

    [Fact]
    public async Task WorkerCrash_FailsOnlyThatImage()
    {
        var files = Photos(2).Append(Photos(1, "crashme")[0]).ToList();
        var crashOptions = new WorkerOptions { Environment = new Dictionary<string, string> { ["IQE_TEST_CRASH_MARKER"] = "crashme" } };
        await using var job = new JobOrchestrator(_workspaces, workerFactory: () => new PythonWorkerClient(crashOptions));

        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        var crashed = job.State!.Images.Single(i => i.Location.SourcePath!.Contains("crashme"));
        Assert.Equal((ProcessingStatus.Failed, "worker_crashed", 2), (crashed.Status, crashed.Reason, crashed.Attempts));
        Assert.Equal(2, job.State.Images.Count(i => i.Status == ProcessingStatus.Completed));
        Assert.Equal(2, job.State.Performance.WorkerRestarts);
    }

    [Fact]
    public async Task TwoWorkers_ProcessInParallel()
    {
        var files = Photos(6);
        await using var job = new JobOrchestrator(_workspaces, new ProcessingOptions { WorkerCount = 2 });
        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        Assert.All(job.State!.Images, i => Assert.Equal(ProcessingStatus.Completed, i.Status));
    }

    // ---------------------------------------------------------------- limits and failures

    [Fact]
    public async Task LowMemoryLimit_SkipsImagesWithReason()
    {
        var files = Photos(2);
        await using var job = new JobOrchestrator(_workspaces, new ProcessingOptions { MaxImageMemoryBytes = 100L * 1024 * 1024 });
        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(Timeout);

        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        Assert.All(job.State!.Images, i => Assert.Equal((ProcessingStatus.Skipped, "too_large"), (i.Status, i.Reason)));
        Assert.All(job.State.Images, i => Assert.Contains("MB", i.Error));
    }

    [Fact]
    public async Task NotEnoughDisk_FailsBeforeDoingAnything()
    {
        await using var job = new JobOrchestrator(_workspaces, new ProcessingOptions { MinFreeDiskBytes = long.MaxValue / 4 });
        var outcome = await job.RunAsync(new JobRequest([_fx.Pdf])).WaitAsync(Timeout);

        Assert.Equal((JobOutcomeKind.Failed, "disk_full"), (outcome.Kind, outcome.ErrorCode));
        Assert.Empty(job.State!.Images);
    }

    [Fact]
    public async Task PasswordProtectedPdf_StopsResumably_ThenContinuesWithPassword()
    {
        JobWorkspace workspace;
        await using (var job = new JobOrchestrator(_workspaces))
        {
            var outcome = await job.RunAsync(new JobRequest([_fx.SecretPdf])).WaitAsync(Timeout);
            Assert.Equal((JobOutcomeKind.Failed, "password_required"), (outcome.Kind, outcome.ErrorCode));
            Assert.NotEqual(JobStage.Failed, job.State!.Stage); // waiting for the password, not finished
            workspace = job.Workspace!;
        }

        var saved = await JobStateStore.LoadAsync(workspace);
        Assert.DoesNotContain("s3cret", System.Text.Encoding.UTF8.GetString(JobStateStore.Serialize(saved!)));

        await using (var wrong = new JobOrchestrator(_workspaces))
        {
            var outcome = await wrong.ResumeAsync(workspace, "nope").WaitAsync(Timeout);
            Assert.Equal("wrong_password", outcome.ErrorCode);
        }

        await using var retry = new JobOrchestrator(_workspaces);
        var ok = await retry.ResumeAsync(workspace, "s3cret").WaitAsync(Timeout);
        Assert.Equal(JobOutcomeKind.ReadyForReview, ok.Kind);
        Assert.Null(retry.State!.ErrorCode);
        Assert.NotEmpty(retry.State.Images);
    }

    [Theory]
    [InlineData("broken.pdf", "%PDF-1.7 not really", "corrupt_pdf")]
    [InlineData("old.doc", "ÐÏ\u0011à¡±\u001Aá", "legacy_doc")] // OLE compound file signature
    public async Task UnreadableDocuments_FailWithExplanation(string name, string content, string code)
    {
        var path = Input(name);
        File.WriteAllBytes(path, content.Select(c => (byte)c).Concat(new byte[600]).ToArray());
        await using var job = new JobOrchestrator(_workspaces);
        var outcome = await job.RunAsync(new JobRequest([path])).WaitAsync(Timeout);

        Assert.Equal((JobOutcomeKind.Failed, code), (outcome.Kind, outcome.ErrorCode));
        Assert.False(string.IsNullOrWhiteSpace(outcome.Message));
    }

    [Fact]
    public async Task MixedDocumentAndImages_AreRejected()
    {
        await using var job = new JobOrchestrator(_workspaces);
        var outcome = await job.RunAsync(new JobRequest([_fx.Pdf, Path.Combine(_fx.Images, "photo.png")]));
        Assert.Equal((JobOutcomeKind.Failed, "unsupported_input"), (outcome.Kind, outcome.ErrorCode));
    }
}
