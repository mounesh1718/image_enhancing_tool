using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>The per-image report: re-altered vs unchanged, steps, quality before/after and accuracy.</summary>
public class ReportTests : IClassFixture<OrchestratorFixture>, IDisposable
{
    private readonly OrchestratorFixture _fx;
    private readonly TempDirectory _root = new();

    public ReportTests(OrchestratorFixture fixture) => _fx = fixture;

    public void Dispose() => _root.Dispose();

    private static ImageJob Job(string id, ProcessingStatus status) => new()
    {
        Id = id,
        DocumentId = "doc",
        SourceType = SourceType.Image,
        Location = new ImageLocation { SourcePath = id + ".png" },
        ResourceRef = id,
        ResourceId = id,
        Status = status,
    };

    [Theory]
    [InlineData(0.99, 0.02, AccuracyRating.High)]
    [InlineData(0.90, 0.20, AccuracyRating.Good)]
    [InlineData(0.70, 0.05, AccuracyRating.Check)]
    [InlineData(0.99, 0.50, AccuracyRating.Check)]
    public void Rate_UsesEdgesKeptAndNewEdges(double kept, double added, AccuracyRating expected)
    {
        var job = Job("a", ProcessingStatus.Completed);
        job.Fidelity = new FidelityMeasures(0.9, kept, added, 1);
        Assert.Equal(expected, ReportBuilder.Rate(job));
    }

    [Fact]
    public void Rate_FlaggedResultIsNeverHigh()
    {
        var job = Job("a", ProcessingStatus.Completed);
        job.Fidelity = new FidelityMeasures(0.99, 0.99, 0.01, 0);
        job.ReviewReasons.Add("heavy_blur");
        Assert.Equal(AccuracyRating.Good, ReportBuilder.Rate(job));
    }

    [Fact]
    public void TextCheck_ChangedTextIsRatedCheck_AndDescribed()
    {
        var job = Job("a", ProcessingStatus.Completed);
        job.Fidelity = new FidelityMeasures(0.99, 0.99, 0.01, 0);
        job.TextCheck = new TextCheckResult(true, false, 120, 0, [], [], 0.9, 0.92);
        Assert.Equal(AccuracyRating.High, ReportBuilder.Rate(job));
        Assert.Equal("Text check: passed · 120 of 120 characters read the same before and after", Wording.TextCheck(job.TextCheck));

        job.TextCheck = new TextCheckResult(true, true, 120, 1, ["452[1] → 452[7]"], ["452[1] → 452[7]"], 0.9, 0.9);
        Assert.Equal(AccuracyRating.Check, ReportBuilder.Rate(job));
        Assert.Contains("numbers: 452[1] → 452[7]", Wording.TextCheck(job.TextCheck));
        Assert.Equal("Text check: no text found", Wording.TextCheck(new TextCheckResult(false, false, 0, 0, [], [], null, null)));
        Assert.Equal("", Wording.TextCheck(null));
    }

    [Fact]
    public void Redrawn_IsRatedVerified_AndReportedWithItsIdentity()
    {
        var job = Job("structure", ProcessingStatus.Completed);
        job.Operations.Add("redraw");
        job.Settings = job.Settings with { Redraw = true };
        job.Redraw = new RedrawResult(true, [], "CC(C)(S)[C@@H](N)C(=O)O", "VVNCNSJFMMFHPL-VKHMYHEASA-N", "C5H11NO2S", 149.21,
            ["C4 S"], 0.9, null);
        Assert.Equal(AccuracyRating.Verified, ReportBuilder.Rate(job));
        Assert.StartsWith("Redrawn (verified by the automatic checks, not by a chemist yet): C5H11NO2S · 149.21 g/mol · stereocentres C4 S",
            Wording.Redraw(job.Redraw));

        var notRedrawn = Job("label", ProcessingStatus.Completed);
        notRedrawn.Classification = Classified.As(ImageCategory.Chemical); // chemical errors are shown for chemical pictures
        notRedrawn.Operations.Add("sharpen");
        notRedrawn.Redraw = new RedrawResult(false, ["stereo marks in the picture do not match the reading"], null, null, null, null, [], 0.8, null, IsChemical: true);
        Assert.StartsWith("Not redrawn: stereo marks", Wording.Redraw(notRedrawn.Redraw));

        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "s.png", Images = [job, notRedrawn] }, _root.Path);
        Assert.Contains("1</b><span>redrawn (chemical structures)", html);
        Assert.Contains("VVNCNSJFMMFHPL-VKHMYHEASA-N", html);
        Assert.Contains("Why it was not redrawn:", html);
        Assert.Contains("stereo marks in the picture do not match the reading", html);
        Assert.Contains("Accuracy: Verified", html);
        Assert.Contains("do not prove that it matches the original", html);
        Assert.DoesNotContain("proven identical", html);
    }

    [Fact]
    public void Mismatches_AreGroupedNumberedAndPlaced()
    {
        var job = Job("structure", ProcessingStatus.Completed);
        job.Classification = Classified.As(ImageCategory.Chemical);
        job.Operations.Add("sharpen");
        var unread = "29 % of the drawing is not explained by what the AI read.";
        job.Redraw = new RedrawResult(false, ["technical reason"], null, null, null, null, [], 0.8, null,
            Findings:
            [
                new RedrawFinding("The drawing has 1 dashed wedge(s) (3D bonds), but the AI read 0.", [10, 20, 60, 50]),
                new RedrawFinding(unread, [100, 100, 140, 120]),
                new RedrawFinding(unread, [200, 100, 240, 120]),
                new RedrawFinding("The AI's two descriptions of the molecule disagree.", null),
            ], IsChemical: true);
        var items = Wording.Mismatches(job);
        Assert.Equal(3, items.Count);                       // the repeated sentence is listed once
        Assert.Equal(2, items[1].Boxes.Count);              // ... and marked in both places
        Assert.Equal([1, 2, 3], items.Select(i => i.Number));
        Assert.Empty(items[2].Boxes);                       // a sentence without a place
        var row = new ViewModels.ImageItemViewModel(job, 0, _root.Path);
        Assert.StartsWith("1. The drawing has", row.MismatchText);
        Assert.Contains("2. " + unread + " (marked 2×)", row.MismatchText);
        Assert.Equal("Structure checks · 3 items to review", row.MismatchSummary);
        Assert.Equal(3, row.MismatchMarks.Count);
        Assert.Equal(new Avalonia.Rect(10, 20, 50, 30), row.MismatchMarks[0].Box);
        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "s.png", Images = [job] }, _root.Path);
        Assert.Contains("Why it was not redrawn:", html);
        Assert.Contains("dashed wedge", html);

        // A sheet of panels: each mismatch says which panel it is in.
        var sheet = Job("sheet", ProcessingStatus.Completed);
        sheet.Operations.AddRange(["panels", "clean_background"]);
        sheet.Panels = [new PanelResult(2, "text_or_diagram", ["clean_background"],
            new RedrawResult(false, ["x"], null, null, null, null, [], 0.5, null, Findings: [new RedrawFinding("No line is drawn here.", [5, 5, 9, 9])], IsChemical: true))];
        Assert.Equal("Panel 2: No line is drawn here.", Wording.Mismatches(sheet)[0].Message);

        // A proven redraw has nothing to report.
        var proven = Job("ok", ProcessingStatus.Completed);
        proven.Operations.Add("redraw");
        proven.Redraw = new RedrawResult(true, [], "C", "K", "CH4", 16, [], 0.9, null);
        Assert.Empty(Wording.Mismatches(proven));
    }

    [Fact]
    public void UnprovenRedraw_IsRatedCheck_AndSaysWhy()
    {
        var job = Job("caffeine", ProcessingStatus.Completed);
        job.Operations.Add("redraw");
        job.Settings = job.Settings with { Redraw = true, RedrawAll = true };
        job.Redraw = new RedrawResult(true, [], "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "C8H10N4O2", 194.19,
            [], 0.85, null, Proven: false,
            Unproven: ["not recognised as a chemical structure with enough confidence (overall 0.85, weakest atom 0.64)",
                       "bond C–N read as single bond does not look like one", "26 % of the drawing is not explained by the reading"]);
        Assert.True(job.IsUnprovenRedraw);
        Assert.Equal(AccuracyRating.Check, ReportBuilder.Rate(job));
        var text = Wording.Redraw(job.Redraw);
        Assert.StartsWith("Redrawn, NOT VERIFIED: C8H10N4O2", text);
        Assert.Contains("(and 1 more)", text);
        Assert.Contains("Compare every atom and bond", text);
        Assert.Contains("compare every atom", Wording.ReviewReason("unproven_redraw"));

        // The review screen must still show every exact failed check when a worker result has no positioned
        // findings (for example, an older saved job or a whole-structure confidence failure).
        var mismatches = Wording.Mismatches(job);
        Assert.Equal(3, mismatches.Count);
        Assert.Contains("not recognised as a chemical structure with enough confidence", mismatches[0].Message);
        var item = new ViewModels.ImageItemViewModel(job, 0, _root.Path);
        Assert.Contains("could not be fully verified", item.ReviewSummary);
        Assert.Contains("single bond does not look like one", item.MismatchText);
        Assert.Equal("Structure checks · 3 items to review", item.MismatchSummary);

        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "s.png", Images = [job] }, _root.Path);
        Assert.Contains("NOT verified: compare every atom and bond", html);
        Assert.Contains("redraw of every chemical structure the reader could read", html);
        Assert.Contains("Not verified because", html);
    }

    [Fact]
    public async Task RedrawJob_RedrawsAKnownStructure_AtTheSamePixelSize()
    {
        Assert.SkipUnless(TestSamples.RedrawInstalled(), "redraw is not installed for the worker's Python");
        var source = Path.Combine(_root.Path, "d-penicillamine.png");
        TestSamples.StructureDrawing(source, "CC(C)(S)[C@@H](N)C(=O)O");
        var options = new ProcessingOptions { WorkerCount = 1, DefaultSettings = new EnhancementSettings { Redraw = true } };
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")), options);
        var outcome = await job.RunAsync(new JobRequest([source])).WaitAsync(TimeSpan.FromMinutes(5));
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);

        var image = Assert.Single(job.SnapshotImages());
        Assert.True(image.IsRedrawn, Wording.Redraw(image.Redraw));
        Assert.Equal("VVNCNSJFMMFHPL-VKHMYHEASA-N", image.Redraw!.InChIKey); // D-penicillamine, (S)
        Assert.Equal(image.PixelSize, image.OutputPixelSize);
        Assert.True(image.NeedsReview); // a chemist must approve every redraw
        Assert.False(ApprovalRules.IsEligibleForBulkApproval(image));
    }

    [Fact]
    public void AiRepair_ChoiceAndDevicesAreDescribed_InDetailsAndReport()
    {
        var job = Job("scan", ProcessingStatus.Completed);
        job.Operations.AddRange(["ai_grain", "clean_background", "sharpen"]);
        job.Fidelity = new FidelityMeasures(0.95, 0.97, 0.03, 4);
        job.Choice = new VersionChoice("ai_clean",
        [
            new VersionCandidate("filters", 15.8, 72.3, null),
            new VersionCandidate("ai_clean", 0, 71.9, null),
            new VersionCandidate("ai_deblur", 0, 80.1, "shows digits the original does not show readably"),
        ], false);
        job.AiDevices["SCUNet"] = "processor";
        job.AiDevices["DocRes"] = "NVIDIA GeForce RTX 3050 Laptop GPU";

        Assert.Equal("Best of 3 versions: AI repair kept (as readable, least grain). Not used: AI text de-blur "
                     + "(shows digits the original does not show readably).", Wording.Choice(job.Choice));
        Assert.Equal("AI ran on: SCUNet – processor, DocRes – NVIDIA GeForce RTX 3050 Laptop GPU", Wording.Devices(job.AiDevices));
        Assert.Equal("AI grain removal", Wording.Operation("ai_grain"));
        Assert.Contains("every letter and digit", Wording.ReviewReason("ai_deblurred_text"));
        Assert.Equal("", Wording.Choice(null));

        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "s.png", Images = [job] }, _root.Path);
        Assert.Contains("Version kept (AI repair)", html);
        Assert.Contains("AI ran on: SCUNet", html);

        var copy = job.Clone();
        copy.AiDevices["FBCNN"] = "processor";
        Assert.False(job.AiDevices.ContainsKey("FBCNN")); // clones do not share the device list
    }

    [Fact]
    public async Task AiRepairJob_CleansGrainyText_AndReportsTheChoice()
    {
        Assert.SkipUnless(TestSamples.AiRepairInstalled(), "AI repair models are not installed for the worker's Python");
        var source = Path.Combine(_root.Path, "grainy label.png");
        TestSamples.GrainyText(source);
        var options = new ProcessingOptions { WorkerCount = 1, DefaultSettings = new EnhancementSettings { AiRepair = true } };
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")), options);
        var outcome = await job.RunAsync(new JobRequest([source])).WaitAsync(TimeSpan.FromMinutes(5));
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);

        var image = Assert.Single(job.SnapshotImages());
        Assert.Equal("ai_clean", image.Choice?.Chosen);
        Assert.Equal("ai_grain", image.Operations[0]);
        Assert.Contains("SCUNet", image.AiDevices.Keys);
        Assert.Equal(image.PixelSize, image.OutputPixelSize);
        Assert.False(image.TextCheck?.Flag ?? false); // the text reads the same as in the original
    }

    [Fact]
    public void Panels_AreDescribedOneByOne_AndASheetIsNotOneRedraw()
    {
        var job = Job("sheet", ProcessingStatus.Completed);
        job.Operations.AddRange(["panels", "redraw", "ai_photo", "clean_background"]);
        job.Panels =
        [
            new PanelResult(1, "text_or_diagram", ["redraw"], new RedrawResult(true, [], "c1ccccc1", "UHOVQNZJYSORNB-UHFFFAOYSA-N", "C6H6", 78.11, [], 0.9, null)),
            new PanelResult(2, "photo", ["ai_photo", "local_contrast"], null),
            new PanelResult(3, "text_or_diagram", [], null),
        ];
        Assert.False(job.IsRedrawn);
        Assert.Equal("Panel 1 (text / diagram): redrawn C6H6", Wording.Panel(job.Panels[0]));
        Assert.Equal("Panel 2 (photo): AI clarity, clarity", Wording.Panel(job.Panels[1]));
        Assert.Equal("Panel 3 (text / diagram): already good, unchanged", Wording.Panel(job.Panels[2]));
        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "sheet.png", Images = [job] }, _root.Path);
        Assert.Contains("UHOVQNZJYSORNB-UHFFFAOYSA-N", html);
        Assert.Contains("Panel 2", html);
    }

    [Fact]
    public void Levels_HaveNamesAndOrder()
    {
        Assert.Equal(["Light", "Standard", "Strong", "Maximum"], Wording.LevelKeys.Select(Wording.LevelName));
        Assert.Equal("maximum", Wording.LevelKey("Maximum"));
        Assert.Equal("Level 1 · Light", Wording.Level("gentle"));
        Assert.Null(Wording.BoldValue(Wording.BoldChoices[0]));
        Assert.True(Wording.BoldValue("On"));
        Assert.False(Wording.BoldValue("Off"));
        Assert.Equal("bolder text", Wording.Operation("bolden"));
        Assert.Equal("AI clarity", Wording.Operation("ai_photo"));
        Assert.Contains("compare it zoomed in", Wording.ReviewReason("ai_restored"));
        Assert.Contains("chemist", Wording.ReviewReason("redrawn_structure"));
    }

    [Fact]
    public void Build_ListsEveryImage_AndEscapesText()
    {
        var done = Job("img-1", ProcessingStatus.Completed);
        done.Operations.AddRange(["denoise", "sharpen"]);
        done.QualityBefore = new QualityMeasures(10, 8, 2, 120);
        done.QualityAfter = new QualityMeasures(20, 2, 1, 200);
        done.Fidelity = new FidelityMeasures(0.93, 0.98, 0.04, 2.5);
        var skipped = Job("img-2", ProcessingStatus.Skipped);
        skipped.Error = "vector graphic <svg>";
        var state = new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "a&b.png", Images = [done, skipped] };

        var html = ReportBuilder.Build(state, _root.Path);

        Assert.Contains("a&amp;b.png", html);
        Assert.Contains("Re-altered", html);
        Assert.Contains("Kept unchanged", html);
        Assert.Contains("Vector graphic &lt;svg&gt;", html);
        Assert.Contains("Noise reduction", html);
        Assert.Contains("Accuracy: High", html);
        Assert.Contains("98.0 %", html);
        Assert.Contains("redrawn", html);
        Assert.Contains("Level 2 · Standard", html);
        Assert.DoesNotContain("<svg>", html);
    }

    [Fact]
    public async Task Build_FromRealJob_HasMeasurementsAndThumbnails()
    {
        var workspaces = new WorkspaceManager(Path.Combine(_root.Path, "jobs"));
        await using var job = new JobOrchestrator(workspaces);
        var outcome = await job.RunAsync(new JobRequest([_fx.Pdf])).WaitAsync(TimeSpan.FromMinutes(3));
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);

        var images = job.SnapshotImages();
        var completed = images.Where(i => i.Status == ProcessingStatus.Completed).ToList();
        Assert.NotEmpty(completed);
        Assert.All(completed, i =>
        {
            Assert.NotNull(i.QualityBefore);
            Assert.NotNull(i.QualityAfter);
            Assert.NotNull(i.Fidelity);
            Assert.InRange(i.Fidelity!.EdgesKept, 0, 1);
        });

        var html = ReportBuilder.Build(job.State!, job.Workspace!.RootPath);
        Assert.Contains("data:image/jpeg;base64,", html);
        Assert.Equal(images.Count, html.Split("<article").Length - 1);
    }
}
