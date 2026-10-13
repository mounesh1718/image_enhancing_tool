using System.Text.Json;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;
using ImageQualityEnhancer.ViewModels;

namespace ImageQualityEnhancer.Tests;

public class ChemicalWarningsTests
{
    private static ImageJob Image() => new()
    {
        Id = "table", DocumentId = "doc", ResourceId = "r1", ResourceRef = "r1",
        SourceType = SourceType.Image, Location = new ImageLocation { SourcePath = "table.png" },
        Status = ProcessingStatus.Completed, NeedsReview = true,
        Operations = ["clean_background"], ReviewReasons = ["text_or_diagram", "redrawn_structure", "unproven_redraw"],
    };

    private static RedrawResult OrdinaryContent() => new(false, ["not recognised as a chemical structure"],
        null, null, null, null, [], 0.1, null,
        Findings: [new RedrawFinding("The AI is not sure about this atom (Control).", [10, 10, 30, 30])], IsChemical: false);

    [Fact]
    public void OrdinaryImage_HidesChemicalDetailsAndStaleWarnings_ButKeepsNormalReview()
    {
        var job = Image();
        job.Redraw = OrdinaryContent();
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.False(row.HasMismatches);
        Assert.False(row.HasMismatchSummary);
        Assert.Empty(row.MismatchMarks);
        Assert.False(row.HasRedraw);
        Assert.Contains("Text or diagram", row.Explanation);
        Assert.DoesNotContain("chemist", row.Explanation);
        Assert.DoesNotContain("atom", row.Explanation);
        var html = ReportBuilder.Build(new JobState { JobId = "j", SourceType = SourceType.Image, SourceFileName = "table.png", Images = [job] }, Path.GetTempPath());
        Assert.DoesNotContain("Not redrawn", html);
        Assert.DoesNotContain("Control", html);
    }

    [Fact]
    public void MixedSheet_ShowsOnlyChemicalPanels_AndKeepsTheirErrorLocations()
    {
        var job = Image();
        job.Operations = ["panels", "clean_background", "redraw"];
        var chemical = new RedrawResult(true, [], "CCO", "key", "C2H6O", 46, [], 0.8, null, Proven: false,
            Unproven: ["a bond could not be verified"], Findings: [new RedrawFinding("Check the C-O bond.", [100, 50, 150, 90])],
            IsChemical: true);
        job.Panels = [new PanelResult(1, "text_or_diagram", ["clean_background"], OrdinaryContent()),
            new PanelResult(2, "text_or_diagram", ["redraw"], chemical)];
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.Equal("Panel 2: Check the C-O bond.", Assert.Single(Wording.Mismatches(job)).Message);
        Assert.Single(row.MismatchMarks);
        Assert.Contains("chemist", row.Explanation);
        Assert.DoesNotContain("not recognised", row.RedrawText);
    }

    [Fact]
    public void AllOrdinaryPanels_HaveNoChemicalWarning_IncludingAfterSaving()
    {
        var job = Image();
        job.Operations = ["panels", "clean_background"];
        job.Panels = [new PanelResult(1, "text_or_diagram", ["clean_background"], OrdinaryContent())];
        var saved = JsonSerializer.Serialize(job, JobStateStore.JsonOptions);
        var restored = JsonSerializer.Deserialize<ImageJob>(saved, JobStateStore.JsonOptions)!;
        var row = new ImageItemViewModel(restored, 0, Path.GetTempPath());
        Assert.False(row.HasMismatches);
        Assert.False(row.HasMismatchSummary);
        Assert.DoesNotContain("not redrawn", row.RedrawText);
        Assert.DoesNotContain("chemist", row.Explanation);
    }

    [Fact]
    public void ActualRedraw_WarningsAreNeverHidden_EvenWithInconsistentClassification()
    {
        var job = Image();
        job.Operations = ["redraw"];
        job.Redraw = OrdinaryContent() with { Redrawn = true, Proven = false, Unproven = ["reading differs"] };
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.True(row.HasMismatches);
        Assert.Contains("chemist", row.Explanation);
        Assert.Contains("not verified", row.StatusText);
        Assert.Contains("could not be fully verified", row.ReviewSummary);
        Assert.Contains("chemist", row.ReviewSummary);
    }

    [Theory]
    [InlineData("photo", true)]
    [InlineData("photo", null)]
    [InlineData("text_or_diagram", null)]
    public void PhotoOrLegacyUnclassifiedGuess_DoesNotShowAtomErrors(string kind, bool? isChemical)
    {
        var job = Image();
        job.ContentKind = kind;
        job.Redraw = OrdinaryContent() with { IsChemical = isChemical, Confidence = 0.4,
            Findings = [new RedrawFinding("The AI is not sure about this atom (Ag, 40 % sure).", [10, 10, 30, 30])] };
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.False(row.HasMismatches);
        Assert.False(row.HasMismatchMarks);
        Assert.False(row.HasRedraw);
        Assert.DoesNotContain("chemist", row.ReviewSummary);
        Assert.DoesNotContain("atom", row.ReviewSummary);
        Assert.DoesNotContain("redraw", row.ReviewSummary);
    }

    [Fact]
    public void KnownChemicalCandidate_KeepsDetailedChecks_AndSimpleSummary()
    {
        var job = Image();
        job.ContentKind = "text_or_diagram";
        job.Classification = Classified.As(ImageCategory.Chemical);
        job.Redraw = OrdinaryContent() with { IsChemical = true };
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.True(row.HasMismatches);
        Assert.True(row.HasMismatchMarks);
        Assert.StartsWith("Structure checks", row.MismatchSummary);
        Assert.Contains("original drawing was enhanced", row.ReviewSummary);
        Assert.DoesNotContain("Control", row.ReviewSummary);
        Assert.Contains("Control", row.MismatchText);
    }

    [Fact]
    public void LegacyMolecularIdentity_KeepsChecks_AndTextChangesStayVisible()
    {
        var job = Image();
        job.Redraw = OrdinaryContent() with { IsChemical = null, InChIKey = "legacy-molecule" };
        var row = new ImageItemViewModel(job, 0, Path.GetTempPath());
        Assert.True(row.HasMismatches);
        job.Redraw = null;
        job.ReviewReasons = ["text_changed"];
        Assert.Contains("Check the text and numbers carefully", row.ReviewSummary);
        Assert.Contains("reads some letters or numbers differently", row.Explanation);
    }
}
