using System.Security.Cryptography;
using System.Text.Json;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>
/// Optional AI enhancement after the normal (OpenCV) result: suggestions, trying AI on one picture or on the whole job,
/// choosing the version to keep, and what approval, export and the report do with that choice.
/// </summary>
public class AiEnhancementTests : IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromMinutes(10);
    private readonly TempDirectory _root = new();

    public void Dispose() => _root.Dispose();

    private static readonly AiSuggestion Grain = new("grain", "SCUNet", "Grain detected — SCUNet may help.", true);

    private static ImageJob Job(string id, ProcessingStatus status = ProcessingStatus.Completed, params AiSuggestion[] suggestions) => new()
    {
        Id = id, DocumentId = "doc", ResourceId = id, ResourceRef = id, SourceType = SourceType.Image,
        Location = new ImageLocation { SourcePath = id + ".png" }, Status = status, EnhancedPath = $"enhanced/{id}.png",
        Operations = ["denoise", "sharpen"], AiSuggestions = [.. suggestions],
    };

    private static AiResult Accepted(string path = "enhanced/photo_ai.png") =>
        new("accepted", ["SCUNet"], path, "thumbnails/photo_ai.jpg", [], ["ai_grain", "sharpen"], new FidelityMeasures(0.97, 0.98, 0.02, 1), null, null);

    private static AiResult Rejected() =>
        new("rejected", ["SCUNet"], "enhanced/photo_ai.png", null, ["SCUNet: more detail lost or added than with the filters"], ["ai_grain"], null, null, null);

    private sealed class Steps : IProgress<(int Done, int Total)>
    {
        public List<(int Done, int Total)> Seen { get; } = [];

        public void Report((int Done, int Total) value)
        {
            lock (Seen)
            {
                Seen.Add(value);
            }
        }
    }

    private static byte[] Sha(string path) => SHA256.HashData(File.ReadAllBytes(path));

    // ---------------------------------------------------------------- without the worker

    [Fact]
    public void WorkerResults_AreRead_SuggestionsAndTheCheckedAiVersion()
    {
        var suggestions = JobOrchestrator.ParseSuggestions(JsonDocument.Parse("""
            {"status": "completed", "aiSuggestions": [
              {"kind": "grain", "model": "SCUNet", "text": "Grain detected — SCUNet may help.", "installed": true},
              {"kind": "jpeg", "model": "FBCNN", "text": "Strong JPEG compression detected — FBCNN may help.", "installed": false}]}
            """).RootElement);
        Assert.Equal([Grain, new AiSuggestion("jpeg", "FBCNN", "Strong JPEG compression detected — FBCNN may help.", false)], suggestions);
        Assert.Empty(JobOrchestrator.ParseSuggestions(JsonDocument.Parse("""{"status": "skipped"}""").RootElement));

        var ai = JobOrchestrator.ParseAiResult(JsonDocument.Parse("""
            {"status": "accepted", "models": ["DocRes"], "reasons": [],
             "operations": [{"op": "ai_deblur_text"}, {"op": "clean_background"}],
             "output": {"path": "C:/job/enhanced/scan_ai.png"}, "thumbnail": {"path": "C:/job/thumbnails/scan_ai.jpg"},
             "checks": {"text": {"textFound": true, "flag": false},
                        "detail": {"ssim": 0.93, "lostEdgeRatio": 0.04, "newEdgeRatio": 0.06, "meanShift": 2.1},
                        "chemical": "MolScribe reads the same structure before and after"}}
            """).RootElement, p => "rel:" + Path.GetFileName(p));
        Assert.True(ai.Accepted);
        Assert.Equal("DocRes", ai.ModelNames);
        Assert.Equal("rel:scan_ai.png", ai.Path);
        Assert.Equal("rel:scan_ai.jpg", ai.ThumbnailPath);
        Assert.Equal(["ai_deblur_text", "clean_background"], ai.Operations);
        Assert.True(ai.TextPassed);
        Assert.Equal(0.96, ai.Fidelity!.EdgesKept, 3);
        Assert.Equal("MolScribe reads the same structure before and after", ai.ChemicalNote);

        var missing = JobOrchestrator.ParseAiResult(JsonDocument.Parse("""
            {"status": "unavailable", "reasons": ["SCUNet is not installed (run worker/setup_ai.py), so it could not be tried."]}
            """).RootElement, p => p);
        Assert.False(missing.Accepted);
        Assert.Null(missing.Path);
        Assert.Equal("AI", missing.ModelNames);
    }

    [Fact]
    public void Eligibility_NeedsASuitableInstalledModel_AndAFinishedPictureNotTriedYet()
    {
        Assert.True(JobOrchestrator.IsAiEligible(Job("a", ProcessingStatus.Completed, Grain)));
        var good = Job("good", ProcessingStatus.Skipped, Grain);
        good.Reason = "already_good";
        Assert.True(JobOrchestrator.IsAiEligible(good)); // nothing to fix with filters, but grain AI may still help

        Assert.False(JobOrchestrator.IsAiEligible(Job("none")));                                              // no suitable repair
        Assert.False(JobOrchestrator.IsAiEligible(Job("missing", ProcessingStatus.Completed, Grain with { Installed = false })));
        Assert.False(JobOrchestrator.IsAiEligible(Job("failed", ProcessingStatus.Failed, Grain)));
        var tried = Job("tried", ProcessingStatus.Completed, Grain);
        tried.Ai = Rejected();
        Assert.False(JobOrchestrator.IsAiEligible(tried));                                                    // tried once on this result
        var sheet = Job("sheet", ProcessingStatus.Completed, Grain);
        sheet.Operations.Add("panels");
        Assert.False(JobOrchestrator.IsAiEligible(sheet));
        var redrawn = Job("redrawn", ProcessingStatus.Completed, Grain);
        redrawn.Operations.Add("redraw");
        Assert.True(redrawn.IsRedrawn);
        Assert.False(JobOrchestrator.IsAiEligible(redrawn));                                                  // never AI on a redraw
    }

    [Fact]
    public void Wording_SuggestsWithoutPromising_AndExplainsEachOutcome()
    {
        Assert.Equal("No suitable AI repair detected.", Wording.AiSuggestions(Job("clean")));
        Assert.Equal("Grain detected — SCUNet may help.", Wording.AiSuggestions(Job("grain", ProcessingStatus.Completed, Grain)));
        Assert.Equal("Grain detected — SCUNet may help. SCUNet is not installed, so it cannot be tried (run worker/setup_ai.py).",
            Wording.AiSuggestions(Job("missing", ProcessingStatus.Completed, Grain with { Installed = false })));
        var sheet = Job("sheet", ProcessingStatus.Completed, Grain);
        sheet.Operations.Add("panels");
        Assert.Contains("not offered for a sheet", Wording.AiSuggestions(sheet));

        var job = Job("photo", ProcessingStatus.Completed, Grain);
        Assert.Equal("", Wording.AiHeadline(job));
        job.Ai = Accepted();
        job.UseAiVersion = true;
        Assert.Equal("AI enhancement applied (SCUNet)", Wording.AiHeadline(job));
        var applied = Wording.AiOutcome(job);
        Assert.StartsWith("The AI version is now used for this picture. It passed the detail check.", applied);
        Assert.Contains("AI may not be better for every picture", applied);
        Assert.Contains("switch back", applied);
        Assert.Contains("It still needs your approval", applied);
        Assert.DoesNotContain("will improve", applied);
        Assert.Equal("AI enhancement applied (SCUNet). It passed the detail check.", Wording.AiSummary(job));

        job.UseAiVersion = false;
        Assert.Equal("AI version ready (SCUNet) · not in use", Wording.AiHeadline(job));
        Assert.Equal("It passed the detail check. The OpenCV result is used. Choose the AI version to use it instead.", Wording.AiOutcome(job));
        job.Approval = ApprovalStatus.Approved;
        Assert.Contains("You had already approved the OpenCV result, so it stays in use until you choose the AI version (you will be asked first).",
            Wording.AiOutcome(job));
        job.Approval = ApprovalStatus.KeepOriginal;
        Assert.Contains("You chose to keep the original", Wording.AiOutcome(job));
        job.Approval = ApprovalStatus.Undecided;

        job.Ai = Rejected();
        Assert.Equal("AI enhancement not applied (SCUNet)", Wording.AiHeadline(job));
        Assert.Equal("It did not pass the checks: it loses or adds more detail than the OpenCV result. The OpenCV result stays in use.",
            Wording.AiOutcome(job));
        Assert.Equal("AI enhancement not applied (SCUNet): it loses or adds more detail than the OpenCV result.", Wording.AiSummary(job));
        job.Ai = new AiResult("unavailable", [], null, null, ["SCUNet is not installed (run worker/setup_ai.py), so it could not be tried."], [], null, null, null);
        Assert.Equal("AI enhancement could not run", Wording.AiHeadline(job));
        Assert.Equal("SCUNet is not installed (run worker/setup_ai.py), so it could not be tried. The OpenCV result stays in use.", Wording.AiOutcome(job));
        job.Ai = new AiResult("not_eligible", [], null, null, ["No suitable AI repair detected."], [], null, null, null);
        Assert.Equal("AI enhancement not applied", Wording.AiHeadline(job));
        Assert.Equal("No suitable AI repair was detected for this picture. The OpenCV result stays in use.", Wording.AiOutcome(job));
        job.Ai = new AiResult("failed", [], null, null, ["the image processor stopped while trying AI on this picture"], [], null, null, null);
        Assert.Equal("The image processor stopped while trying AI on this picture. The OpenCV result stays in use.", Wording.AiOutcome(job));
        Assert.Equal("AI enhancement could not run: the image processor stopped while trying AI on this picture.", Wording.AiSummary(job));
    }

    [Fact]
    public void ChosenAiVersion_IsTheResult_AndNeedsItsOwnApproval_NeverBulkApproved()
    {
        var job = Job("photo", ProcessingStatus.Completed, Grain);
        Assert.Equal("enhanced/photo.png", job.ResultPath);
        job.Ai = Accepted();
        Assert.Equal("enhanced/photo.png", job.ResultPath); // trying AI changes nothing until the user chooses it
        Assert.True(ApprovalRules.IsEligibleForBulkApproval(job));

        job.UseAiVersion = true;
        Assert.Equal("enhanced/photo_ai.png", job.ResultPath);
        Assert.True(ApprovalRules.CanApprove(job));
        Assert.False(ApprovalRules.IsEligibleForBulkApproval(job)); // the user approves an AI version picture by picture

        // AI on a picture filters left unchanged: the AI file is the result.
        var good = Job("good", ProcessingStatus.Skipped, Grain);
        good.Reason = "already_good";
        good.EnhancedPath = null;
        Assert.False(ApprovalRules.CanApprove(good));
        good.Ai = Accepted("enhanced/good_ai.png");
        good.UseAiVersion = true;
        Assert.Equal("enhanced/good_ai.png", good.ResultPath);
        Assert.True(ApprovalRules.CanApprove(good));
    }

    [Fact]
    public void Report_SaysWhichVersionWasChosen_ForEveryTriedPicture()
    {
        var chosen = Job("photo", ProcessingStatus.Completed, Grain);
        chosen.Ai = Accepted();
        chosen.UseAiVersion = true;
        chosen.Approval = ApprovalStatus.Approved;
        var kept = Job("chart", ProcessingStatus.Completed, Grain);
        kept.Ai = Accepted("enhanced/chart_ai.png");
        var rejected = Job("table", ProcessingStatus.Completed, Grain);
        rejected.Ai = Rejected();
        var untouched = Job("logo");

        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "s.png",
            Images = [chosen, kept, rejected, untouched] }, _root.Path);
        Assert.Contains("AI enhancement (optional, tried after processing): <b>3</b> tried · <b>2</b> usable · <b>1</b> not usable · "
                        + "<b>1</b> chosen instead of the OpenCV result.", html);
        Assert.Contains("<b>Version chosen:</b> the AI version (SCUNet). AI enhancement applied (SCUNet). It passed the detail check.", html);
        Assert.Contains("<b>Version chosen:</b> the OpenCV result. " + System.Net.WebUtility.HtmlEncode("AI version ready (SCUNet) · not in use."), html);
        Assert.Contains("<b>Version chosen:</b> the OpenCV result. AI enhancement not applied (SCUNet): it loses or adds more detail", html);
        Assert.Contains("Process applied (AI version):", html);
        Assert.Equal(3, html.Split("Version chosen:").Length - 1); // not for the picture AI was never tried on
    }

    // ---------------------------------------------------------------- with the worker

    [Fact]
    public async Task ChoosingAVersion_AsksBeforeChangingAnApproval_AndExportSavesTheChosenVersion()
    {
        var photo = Path.Combine(_root.Path, "photo.png");
        TestSamples.NoisyPhoto(photo);
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")));
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await job.RunAsync(new JobRequest([photo])).WaitAsync(Timeout)).Kind);
        var image = job.State!.Images.Single();
        Assert.Equal(ProcessingStatus.Completed, image.Status);
        Assert.Contains(image.AiSuggestions, s => s.Kind == "grain"); // the suggestion comes with the normal result

        // An accepted AI version of the same pixel size (here: the original file, so it differs from the OpenCV result).
        var aiFile = job.Workspace!.Resolve(WorkspaceFolder.Enhanced, image.ResourceId + "_ai.png");
        File.Copy(photo, aiFile);
        image.Ai = Rejected();
        await Assert.ThrowsAsync<InvalidOperationException>(() => job.ChooseVersionAsync(image.Id, useAi: true)); // only an accepted one
        image.Ai = Accepted(job.Workspace.ToRelative(aiFile));

        await job.SetApprovalAsync([image.Id], ApprovalStatus.Approved);
        Assert.False(await job.ChooseVersionAsync(image.Id, useAi: true)); // approved: the user must confirm first
        image = job.SnapshotImages().Single();
        Assert.False(image.UseAiVersion);
        Assert.Equal(ApprovalStatus.Approved, image.Approval);

        Assert.True(await job.ChooseVersionAsync(image.Id, useAi: true, approvalChangeConfirmed: true));
        image = job.SnapshotImages().Single();
        Assert.True(image.UseAiVersion);
        Assert.Equal(ApprovalStatus.Undecided, image.Approval); // the AI version needs its own approval
        Assert.Equal(0, await job.ApproveAllEligibleAsync());    // and never gets it in bulk
        var saved = (await JobStateStore.LoadAsync(job.Workspace))!.Images.Single();
        Assert.True(saved.UseAiVersion);
        Assert.Equal("accepted", saved.Ai!.Status);

        await job.SetApprovalAsync([image.Id], ApprovalStatus.Approved);
        var output = Path.Combine(_root.Path, "out");
        Directory.CreateDirectory(output);
        var result = await job.ExportAsync(output);
        Assert.Equal(1, result.ImagesReplaced);
        Assert.Equal(Sha(aiFile), Sha(Assert.Single(Directory.GetFiles(output, "*.png"))));

        // Back to the OpenCV result: approved, so it asks first; once undecided, switching needs no confirmation.
        Assert.False(await job.ChooseVersionAsync(image.Id, useAi: false));
        Assert.True(await job.ChooseVersionAsync(image.Id, useAi: false, approvalChangeConfirmed: true));
        image = job.SnapshotImages().Single();
        Assert.False(image.UseAiVersion);
        Assert.Equal(image.EnhancedPath, image.ResultPath);
        Assert.True(await job.ChooseVersionAsync(image.Id, useAi: true));
    }

    [Fact]
    public async Task TryAi_OnOnePicture_AppliesItWhenStillToDecide_KeepsAnApproval_AndLeavesIneligiblePicturesAlone()
    {
        Assert.SkipUnless(TestSamples.AiRepairInstalled(), "the AI repair models are not installed (worker/setup_ai.py)");
        var photo = Path.Combine(_root.Path, "photo.png");
        var grainy = Path.Combine(_root.Path, "grainy.png");
        var table = Path.Combine(_root.Path, "table.png");
        TestSamples.NoisyPhoto(photo);
        TestSamples.NoisyPhoto(grainy);
        TestSamples.OrdinaryTable(table);
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")));
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await job.RunAsync(new JobRequest([photo, grainy, table])).WaitAsync(Timeout)).Kind);
        var byName = job.SnapshotImages().ToDictionary(i => Path.GetFileNameWithoutExtension(i.Location.SourcePath!));
        Assert.True(JobOrchestrator.IsAiEligible(byName["photo"]), Wording.AiSuggestions(byName["photo"]));
        Assert.False(JobOrchestrator.IsAiEligible(byName["table"]), Wording.AiSuggestions(byName["table"]));
        Assert.Equal("No suitable AI repair detected.", Wording.AiSuggestions(byName["table"]));
        Assert.All(byName.Values, i => Assert.DoesNotContain(i.Operations, o => o.StartsWith("ai_", StringComparison.Ordinal))); // no AI until asked

        await job.SetApprovalAsync([byName["photo"].Id], ApprovalStatus.Approved);
        var openCv = Path.Combine(job.Workspace!.RootPath, byName["photo"].EnhancedPath!);
        var openCvHash = Sha(openCv);
        var steps = new Steps();

        var summary = await job.TryAiAsync([byName["photo"].Id], steps).WaitAsync(Timeout);

        Assert.Equal(new AiRunSummary(1, summary.Accepted, summary.Rejected, 0, Applied: 0), summary); // approved: not applied
        Assert.Equal([(0, 1), (1, 1)], steps.Seen);
        var after = job.SnapshotImages().ToDictionary(i => Path.GetFileNameWithoutExtension(i.Location.SourcePath!));
        var ai = after["photo"].Ai!;
        Assert.Contains(ai.Status, new[] { "accepted", "rejected" });
        Assert.Equal(["SCUNet"], ai.Models);
        Assert.Equal(summary.Accepted == 1, ai.Accepted);
        Assert.True(File.Exists(Path.Combine(job.Workspace.RootPath, ai.Path!)));
        Assert.NotEqual(after["photo"].EnhancedPath, ai.Path);
        Assert.Equal(openCvHash, Sha(openCv));                            // the OpenCV result is kept as it was
        Assert.Equal(after["photo"].EnhancedPath, after["photo"].ResultPath);
        Assert.False(after["photo"].UseAiVersion);                        // approved: not switched without asking
        Assert.Equal(ApprovalStatus.Approved, after["photo"].Approval);   // the earlier approval is unchanged
        Assert.Null(after["table"].Ai);
        Assert.Equal(ApprovalStatus.Undecided, after["table"].Approval);
        Assert.False(string.IsNullOrEmpty(Wording.AiOutcome(after["photo"])));

        // Asking again, or for an ineligible picture, does nothing.
        Assert.Equal(new AiRunSummary(0, 0, 0, 2), await job.TryAiAsync([after["photo"].Id, after["table"].Id]));
        Assert.Equal(ai, job.SnapshotImages().Single(i => i.Id == after["photo"].Id).Ai);

        // A picture still to decide: a usable AI version is applied at once (not approved), and the OpenCV result stays available.
        var second = await job.TryAiAsync([after["grainy"].Id]).WaitAsync(Timeout);
        var grain = job.SnapshotImages().Single(i => i.Id == after["grainy"].Id);
        Assert.Equal(second.Accepted, second.Applied);
        Assert.Equal(grain.Ai!.Accepted, grain.UseAiVersion);
        Assert.Equal(ApprovalStatus.Undecided, grain.Approval);
        Assert.Equal(grain.UseAiVersion ? grain.Ai.Path : grain.EnhancedPath, grain.ResultPath);
        Assert.True(File.Exists(Path.Combine(job.Workspace.RootPath, grain.EnhancedPath!)));
        if (grain.UseAiVersion)
        {
            await job.ApproveAllEligibleAsync();
            Assert.Equal(ApprovalStatus.Undecided, job.SnapshotImages().Single(i => i.Id == grain.Id).Approval); // never approved in bulk
            Assert.True(await job.ChooseVersionAsync(grain.Id, useAi: false)); // switching back needs no question: still to decide
            Assert.Equal(grain.EnhancedPath, job.SnapshotImages().Single(i => i.Id == grain.Id).ResultPath);
        }
    }

    [Fact]
    public async Task TryAi_OnTheWholeJob_ProcessesEachSharedPictureOnce_WithProgress_AppliesOnlyWhereStillToDecide()
    {
        Assert.SkipUnless(TestSamples.AiRepairInstalled(), "the AI repair models are not installed (worker/setup_ai.py)");
        var images = Path.Combine(_root.Path, "set");
        Directory.CreateDirectory(images);
        TestSamples.ImageSet(images);
        var docx = Path.Combine(_root.Path, "report.docx");
        DocxSamples.Build(docx, images);
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")));
        Assert.Equal(JobOutcomeKind.ReadyForReview, (await job.RunAsync(new JobRequest([docx])).WaitAsync(Timeout)).Kind);

        var before = job.SnapshotImages();
        var groups = before.GroupBy(i => i.ResourceRef).ToList();
        var eligible = groups.Where(g => JobOrchestrator.IsAiEligible(g.First())).Select(g => g.Key).ToHashSet();
        Assert.Contains(groups, g => eligible.Contains(g.Key) && g.Count() > 1); // a picture shown in several places
        Assert.Contains(groups, g => !eligible.Contains(g.Key));                   // and pictures AI is not suited to

        // Approve the shared picture first: trying AI must not change that decision.
        var shared = groups.First(g => eligible.Contains(g.Key) && g.Count() > 1).Key;
        await job.SetApprovalAsync(before.Where(i => i.ResourceRef == shared && ApprovalRules.CanApprove(i)).Select(i => i.Id).ToList(),
            ApprovalStatus.Approved);
        before = job.SnapshotImages();
        var enhancedHashes = before.Where(i => i.EnhancedPath is not null).Select(i => i.EnhancedPath!).Distinct()
            .ToDictionary(p => p, p => Sha(Path.Combine(job.Workspace!.RootPath, p)));
        var steps = new Steps();

        var summary = await job.TryAiAsync(before.Select(i => i.Id).ToList(), steps).WaitAsync(Timeout);

        Assert.Equal(eligible.Count, summary.Tried);                    // each shared picture once
        Assert.Equal(groups.Count - eligible.Count, summary.Ineligible);
        Assert.Equal(Enumerable.Range(0, summary.Tried + 1).Select(d => (d, summary.Tried)), steps.Seen);

        var after = job.SnapshotImages();
        var outcomes = after.Where(i => eligible.Contains(i.ResourceRef)).GroupBy(i => i.ResourceRef).Select(g => g.First().Ai!.Status).ToList();
        Assert.Equal(summary.Accepted, outcomes.Count(s => s == "accepted"));
        Assert.Equal(summary.Rejected, outcomes.Count(s => s == "rejected"));
        var appliedRefs = after.Where(i => i.UseAiVersion).Select(i => i.ResourceRef).ToHashSet();
        Assert.Equal(summary.Applied, appliedRefs.Count);
        Assert.DoesNotContain(shared, appliedRefs); // the approved picture keeps its approved version
        foreach (var group in after.GroupBy(i => i.ResourceRef))
        {
            if (eligible.Contains(group.Key))
            {
                Assert.All(group, i => Assert.NotNull(i.Ai));
                Assert.Single(group.Select(i => i.Ai).Distinct());     // one AI version shared by every place it appears
            }
            else
            {
                Assert.All(group, i => Assert.Null(i.Ai));             // ineligible pictures are left unchanged
            }
        }

        var aiFiles = Directory.GetFiles(Path.GetDirectoryName(job.Workspace!.Resolve(WorkspaceFolder.Enhanced, "any.png"))!, "*_ai.*");
        Assert.Equal(after.Where(i => i.Ai?.Path is not null).Select(i => i.Ai!.Path).Distinct().Count(), aiFiles.Length);
        Assert.True(aiFiles.Length <= summary.Tried);
        foreach (var (b, a) in before.Zip(after))
        {
            Assert.Equal(b.Id, a.Id);
            Assert.Equal(b.Approval, a.Approval);                    // nothing approved or un-approved
            Assert.Equal(b.Status, a.Status);
            Assert.Equal(b.EnhancedPath, a.EnhancedPath);           // the OpenCV result stays available
            Assert.Equal(a.Ai is { Accepted: true } && before.Where(i => i.ResourceRef == a.ResourceRef).All(i => i.Approval == ApprovalStatus.Undecided),
                a.UseAiVersion);                                     // applied only where every place is still to decide
        }

        Assert.All(enhancedHashes, kv => Assert.Equal(kv.Value, Sha(Path.Combine(job.Workspace.RootPath, kv.Key))));
        Assert.Equal(new AiRunSummary(0, 0, 0, groups.Count), await job.TryAiAsync(after.Select(i => i.Id).ToList())); // never twice
    }
}
