using System.Security.Cryptography;
using System.Text.Json;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;
using ImageQualityEnhancer.Services.Docx;

namespace ImageQualityEnhancer.Tests;

/// <summary>Export end to end: process, approve, export, then check the saved output against the original.</summary>
public class ExportTests : IClassFixture<OrchestratorFixture>, IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromMinutes(3);
    private readonly OrchestratorFixture _fx;
    private readonly TempDirectory _root = new();
    private readonly WorkspaceManager _workspaces;

    public ExportTests(OrchestratorFixture fixture)
    {
        _fx = fixture;
        _workspaces = new WorkspaceManager(Path.Combine(_root.Path, "jobs"));
        Directory.CreateDirectory(Out);
    }

    public void Dispose() => _root.Dispose();

    private string Out => Path.Combine(_root.Path, "out");

    private static byte[] Sha(string path) => SHA256.HashData(File.ReadAllBytes(path));

    private async Task<JobOrchestrator> ReadyJob(params string[] files)
    {
        var job = new JobOrchestrator(_workspaces);
        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(Timeout);
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);
        return job;
    }

    /// <summary>Approves every finished image, including flagged ones (as a reviewer would after checking).</summary>
    private static async Task ApproveAll(JobOrchestrator job) =>
        await job.SetApprovalAsync(job.SnapshotImages().Where(ApprovalRules.CanApprove).Select(i => i.Id).ToList(), ApprovalStatus.Approved);

    [Fact]
    public async Task Pdf_ExportWritesValidatedCopy_OriginalUntouched()
    {
        var before = Sha(_fx.Pdf);
        await using var job = await ReadyJob(_fx.Pdf);
        await ApproveAll(job);
        var destination = Path.Combine(Out, "report_enhanced.pdf");

        var result = await job.ExportAsync(destination);

        Assert.True(File.Exists(destination));
        Assert.Equal(before, Sha(_fx.Pdf));
        Assert.Equal(job.SnapshotImages().Count(i => i.Approval == ApprovalStatus.Approved), result.ImagesReplaced);
        Assert.Empty(result.NotReplaced);
        Assert.Equal([destination], Directory.GetFiles(Out)); // no temporary file left next to it
        Assert.Equal(destination, job.State!.LastExport!.Destination);
        Assert.Equal(JobStage.AwaitingReview, job.State.Stage); // can review and export again

        // Independent check with the worker: same pages, unapproved images byte-identical.
        await using var worker = new PythonWorkerClient();
        await worker.StartAsync();
        var replaced = job.SnapshotImages().Where(i => i.Approval == ApprovalStatus.Approved).Select(i => i.Location.Xref!.Value).Distinct();
        var check = await worker.SendAsync("pdf_validate", new { original = _fx.Pdf, output = destination, replaced });
        Assert.True(check.GetProperty("ok").GetBoolean(), check.ToString());
    }

    [Fact]
    public async Task Docx_ExportOnlyApprovedOccurrences()
    {
        await using var job = await ReadyJob(_fx.Docx);
        var inTable = job.SnapshotImages().Single(i => i.Location.InTable);
        await job.SetApprovalAsync([inTable.Id], ApprovalStatus.Approved);
        var destination = Path.Combine(Out, "report_enhanced.docx");

        var result = await job.ExportAsync(destination);

        Assert.Equal(1, result.ImagesReplaced);
        var original = DocxScanner.Scan(_fx.Docx);
        var exported = DocxScanner.Scan(destination);
        Assert.Equal(original.Occurrences.Count, exported.Occurrences.Count);
        var changed = exported.Occurrences.Where((o, i) => o.ImageUri != original.Occurrences[i].ImageUri).ToList();
        Assert.Single(changed); // only the approved placement points to a new image
        Assert.True(changed[0].Ref.InTable);
    }

    [Fact]
    public async Task Images_ExportApprovedFilesWithSuffix_AndReassembleTiffPages()
    {
        var tiff = Path.Combine(_root.Path, "scan pages.tif");
        TestSamples.MultiPageTiff(tiff);
        var photo = Path.Combine(_root.Path, "photo.png");
        File.Copy(Path.Combine(_fx.Images, "photo.png"), photo);
        var hashes = new[] { tiff, photo }.ToDictionary(f => f, Sha);

        await using var job = await ReadyJob(tiff, photo);
        var images = job.SnapshotImages();
        var page2 = images.Single(i => i.Location.SourcePath == tiff && i.Location.FrameIndex == 1);
        var photoJob = images.Single(i => i.Location.SourcePath == photo);
        await job.SetApprovalAsync([page2.Id, photoJob.Id], ApprovalStatus.Approved);

        var result = await job.ExportAsync(Out);

        Assert.Equal(2, result.Files.Count);
        Assert.Contains(Path.Combine(Out, "photo_enhanced.png"), result.Files);
        var tiffOut = Path.Combine(Out, "scan pages_enhanced.tif");
        Assert.Contains(tiffOut, result.Files);
        await using var worker = new PythonWorkerClient();
        await worker.StartAsync();
        Assert.Equal(3, (await worker.SendAsync("probe", new { path = tiffOut })).GetProperty("frames").GetInt32());
        Assert.All(hashes, kv => Assert.Equal(kv.Value, Sha(kv.Key)));

        // Exporting again never overwrites the earlier files.
        var again = await job.ExportAsync(Out);
        Assert.Contains(Path.Combine(Out, "photo_enhanced (2).png"), again.Files);
    }

    [Fact]
    public async Task Export_RefusesToOverwriteOriginal()
    {
        var copy = Path.Combine(_root.Path, "mine.pdf");
        File.Copy(_fx.Pdf, copy);
        var before = Sha(copy);
        await using var job = await ReadyJob(copy);
        await ApproveAll(job);

        var failure = await Assert.ThrowsAsync<JobFailure>(() => job.ExportAsync(copy));
        Assert.Equal("same_as_original", failure.Code);
        Assert.Equal(before, Sha(copy));
    }

    [Fact]
    public async Task Export_WithNothingApproved_IsRefused()
    {
        await using var job = await ReadyJob(_fx.Pdf);
        var failure = await Assert.ThrowsAsync<JobFailure>(() => job.ExportAsync(Path.Combine(Out, "x.pdf")));
        Assert.Equal("nothing_approved", failure.Code);
        Assert.Empty(Directory.GetFiles(Out));
    }

    [Fact]
    public async Task Export_ToUnwritableLocation_LeavesNothingBehind()
    {
        await using var job = await ReadyJob(_fx.Pdf);
        await ApproveAll(job);
        var missingFolder = Path.Combine(_root.Path, "does-not-exist", "out.pdf");

        await Assert.ThrowsAsync<JobFailure>(() => job.ExportAsync(missingFolder));
        Assert.False(File.Exists(missingFolder));
        Assert.Equal(JobStage.AwaitingReview, job.State!.Stage); // still reviewable; nothing was lost
    }

    [Fact]
    public async Task EncryptedPdf_ExportAfterRestartNeedsPassword_AndKeepsEncryption()
    {
        JobWorkspace workspace;
        await using (var first = new JobOrchestrator(_workspaces))
        {
            Assert.Equal(JobOutcomeKind.ReadyForReview, (await first.RunAsync(new JobRequest([_fx.SecretPdf], "s3cret")).WaitAsync(Timeout)).Kind);
            await ApproveAll(first);
            workspace = first.Workspace!;
        }

        // A new session: the password was never saved.
        await using var reopened = new JobOrchestrator(_workspaces);
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await reopened.ResumeAsync(workspace).WaitAsync(Timeout)).Kind);
        var destination = Path.Combine(Out, "secret_enhanced.pdf");
        var failure = await Assert.ThrowsAsync<JobFailure>(() => reopened.ExportAsync(destination));
        Assert.Equal("password_required", failure.Code);
        Assert.False(File.Exists(destination));

        await reopened.ExportAsync(destination, "s3cret");
        await using var worker = new PythonWorkerClient();
        await worker.StartAsync();
        var ex = await Assert.ThrowsAsync<WorkerException>(() => worker.SendAsync("pdf_scan", new { path = destination }));
        Assert.Equal("password_required", ex.Code); // still protected
    }

    [Fact]
    public async Task SignedPdf_IsFlagged()
    {
        var signed = Path.Combine(_root.Path, "signed.pdf");
        TestSamples.SignedPdf(signed);
        await using var job = await ReadyJob(signed);
        Assert.True(job.State!.DigitallySigned);

        await using var plain = await ReadyJob(_fx.Pdf);
        Assert.False(plain.State!.DigitallySigned);
    }

    [Fact]
    public async Task Diagnostics_HaveNumbersButNoDocumentContent_AndFinishRemovesTempFiles()
    {
        var named = Path.Combine(_root.Path, "Confidential_Trial_Results.pdf");
        File.Copy(_fx.Pdf, named);
        await using var job = await ReadyJob(named);
        await ApproveAll(job);
        await job.ExportAsync(Path.Combine(Out, "result.pdf"));

        var diagnostics = job.GetDiagnostics();
        Assert.True(diagnostics.TotalImages > 0);
        Assert.Contains("export", diagnostics.StageSeconds.Keys);
        Assert.True(diagnostics.PeakWorkerMemoryMegabytes > 0);
        var json = await File.ReadAllTextAsync(Path.Combine(job.Workspace!.RootPath, "metadata", "diagnostics.json"));
        Assert.DoesNotContain("Confidential", json);
        Assert.DoesNotContain("Quarterly report", json); // document text

        await job.FinishAsync();
        Assert.False(Directory.Exists(job.Workspace.RootPath));
        Assert.True(File.Exists(Path.Combine(Out, "result.pdf")));
        Assert.True(File.Exists(named));
    }
}
