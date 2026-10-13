using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;
using ImageQualityEnhancer.ViewModels;

namespace ImageQualityEnhancer.Tests;

/// <summary>Suggestions from DocumentFigureClassifier as the worker reports them, for tests without a worker.</summary>
internal static class Classified
{
    public static ImageClassification As(ImageCategory category, double confidence = 0.97, ChemicalCheck? check = null,
        params ClassifiedPanel[] panels) =>
        new(category, confidence, category switch
        {
            ImageCategory.Chemical => "chemistry_structure",
            ImageCategory.Table => "table",
            ImageCategory.FlowchartOrGraph => "flow_chart",
            _ => "photograph",
        }, confidence < 0.6, panels, check, "DocumentFigureClassifier v2.5", null);

    public static ChemicalCheck Read(int atoms = 7, int bonds = 6) => new("read", atoms, bonds, 0.88, "C5H11NO2S", []);

    public static ChemicalCheck Uncertain(string reason = "MolScribe is only 55 % sure of its reading (at least 70 % is needed)") =>
        new("uncertain", 4, 2, 0.55, null, [reason]);
}

/// <summary>Picture groups: the classifier's suggestion, the user's choice, and what chemical pictures show.</summary>
public class CategoryTests : IDisposable
{
    private readonly TempDirectory _root = new();

    public void Dispose() => _root.Dispose();

    private static ImageJob Job(string id, ImageClassification? classification = null) => new()
    {
        Id = id, DocumentId = "doc", ResourceId = id, ResourceRef = id, SourceType = SourceType.Image,
        Location = new ImageLocation { SourcePath = id + ".png" }, Status = ProcessingStatus.Completed,
        Operations = ["sharpen"], Classification = classification,
    };

    [Fact]
    public void WorkerResult_IsReadWithPanelsAndChemicalChecks()
    {
        var json = JsonDocument.Parse("""
            {"available": true, "model": "DocumentFigureClassifier v2.5", "category": "table", "confidence": 0.437,
             "label": "table", "uncertain": true,
             "panels": [
               {"panel": 1, "box": [10, 20, 300, 260], "category": "chemical", "confidence": 0.92, "label": "chemistry_structure",
                "uncertain": false, "chemicalCheck": {"status": "read", "atoms": 7, "bonds": 6, "confidence": 0.88, "formula": "C5H11NO2S", "reasons": []}},
               {"panel": 2, "box": [310, 20, 600, 260], "category": "other", "confidence": 0.99, "label": "photograph", "uncertain": false}
             ]}
            """).RootElement;
        var c = JobOrchestrator.ParseClassification(json);
        Assert.Equal(ImageCategory.Table, c.Category);
        Assert.Equal(0.437, c.Confidence);
        Assert.True(c.Uncertain);
        Assert.Equal("DocumentFigureClassifier v2.5", c.Model);
        Assert.Equal(2, c.Panels.Count);
        Assert.Equal(ImageCategory.Chemical, c.Panels[0].Category);
        Assert.Equal([10, 20, 300, 260], c.Panels[0].Box);
        Assert.True(c.Panels[0].ChemicalCheck!.IsRead);
        Assert.Equal("C5H11NO2S", c.Panels[0].ChemicalCheck!.Formula);
        Assert.Null(c.Panels[1].ChemicalCheck); // MolScribe never runs on a photo panel

        var notRead = JobOrchestrator.ParseClassification(JsonDocument.Parse(
            """{"category": "other", "confidence": null, "panels": [], "note": "not classified: SVG is a vector graphic"}""").RootElement);
        Assert.Null(notRead.Confidence);
        Assert.Equal("Not classified: SVG is a vector graphic.", Wording.ClassificationDetails(Job("v", notRead)));
    }

    [Fact]
    public void Groups_CountSheetsInEachKind_AndTheUsersChoiceReplacesThemAll()
    {
        var sheet = Job("sheet", Classified.As(ImageCategory.Table, 0.44, null,
            new ClassifiedPanel(1, [0, 0, 10, 10], ImageCategory.Chemical, 0.9, "chemistry_structure", false, Classified.Read()),
            new ClassifiedPanel(2, [10, 0, 20, 10], ImageCategory.Other, 0.99, "photograph", false, null)));
        Assert.Equal([ImageCategory.Table, ImageCategory.Chemical, ImageCategory.Other], sheet.Groups);
        Assert.True(sheet.IsChemicalCategory);
        Assert.Equal("Sheet: table, chemical diagram, other image", Wording.CategoryLine(sheet));

        sheet.ManualCategory = ImageCategory.FlowchartOrGraph;
        Assert.Equal([ImageCategory.FlowchartOrGraph], sheet.Groups);
        Assert.False(sheet.IsChemicalCategory);
        Assert.Equal("Flowchart or graph · set by you", Wording.CategoryLine(sheet));

        var unclassified = Job("u");
        Assert.Equal(ImageCategory.Unclassified, unclassified.Category);
        Assert.Equal([ImageCategory.Other], unclassified.Groups); // listed with the other images
        Assert.Equal("Not classified", Wording.CategoryLine(unclassified));
        Assert.Equal("Table · 97 %", Wording.CategoryLine(Job("t", Classified.As(ImageCategory.Table))));
        Assert.Equal("Other image · 55 % (unsure)", Wording.CategoryLine(Job("o", Classified.As(ImageCategory.Other, 0.55))));
    }

    [Fact]
    public void EnhanceRequest_CarriesTheCategories_AndOnlyChemicalOnesAllowRedraw()
    {
        string Json(ImageJob job) => JsonSerializer.Serialize(JobOrchestrator.ClassificationRequest(job));
        var sheet = Job("sheet", Classified.As(ImageCategory.Other, 0.5, null,
            new ClassifiedPanel(1, [0, 0, 10, 10], ImageCategory.Chemical, 0.9, "chemistry_structure", false, null)));
        Assert.Equal("""{"category":"other","panels":[{"box":[0,0,10,10],"category":"chemical"}]}""", Json(sheet));
        sheet.ManualCategory = ImageCategory.Table; // the user's choice applies to every panel
        Assert.Equal("""{"category":"table","panels":[]}""", Json(sheet));
        Assert.Equal("""{"category":"unclassified","panels":[]}""", Json(Job("u")));
    }

    [Fact]
    public void ChemicalStatus_KeepsClassifiedReadAndVerifiedApart()
    {
        var read = Job("read", Classified.As(ImageCategory.Chemical, 0.99, Classified.Read()));
        var status = Wording.ChemicalStatus(read);
        Assert.StartsWith("Classified as a chemical diagram — not verified.", status);
        Assert.Contains("recognised 7 atoms and 6 bonds (C5H11NO2S)", status);
        Assert.Contains("not a verified structure", status);
        Assert.False(read.IsPossibleChemical);
        Assert.False(read.IsVerifiedStructure);

        // Looks chemical, but MolScribe cannot read it reliably: stays chemical, marked for review.
        var unsure = Job("unsure", Classified.As(ImageCategory.Chemical, 0.74, Classified.Uncertain()));
        Assert.True(unsure.IsPossibleChemical);
        Assert.Contains(ImageCategory.Chemical, unsure.Groups);
        Assert.Contains("Possible chemical diagram — needs review", Wording.ChemicalStatus(unsure));
        Assert.Contains("55 % sure", Wording.ChemicalStatus(unsure));
        var row = new ImageItemViewModel(unsure, 0, _root.Path);
        Assert.True(row.IsPossibleChemical);
        Assert.Contains("Possible chemical diagram", row.ReviewSummary);
        Assert.False(ApprovalRules.IsEligibleForBulkApproval(unsure)); // never approved in bulk

        Assert.True(Job("missing", Classified.As(ImageCategory.Chemical, 0.99, new ChemicalCheck("unavailable", 0, 0, 0, null, ["not installed"]))).IsPossibleChemical);

        // Verified: every automatic redraw check passed. Still not proof that it matches the original.
        var verified = Job("verified", Classified.As(ImageCategory.Chemical, 0.99, Classified.Read()));
        verified.Operations = ["redraw"];
        verified.Redraw = new RedrawResult(true, [], "CC", "KEY", "C2H6", 30, [], 0.9, null, IsChemical: true);
        Assert.True(verified.IsVerifiedStructure);
        var verifiedStatus = Wording.ChemicalStatus(verified);
        Assert.StartsWith("Chemical structure verified by the automatic checks", verifiedStatus);
        Assert.Contains("do not prove that the redraw matches the original", verifiedStatus);
        Assert.Contains("chemist", verifiedStatus);
        Assert.DoesNotContain("proven", new ImageItemViewModel(verified, 0, _root.Path).AccuracyText, StringComparison.OrdinalIgnoreCase);
        Assert.False(ApprovalRules.IsEligibleForBulkApproval(verified)); // a redraw is never approved automatically
    }

    [Fact]
    public void OnlyDoubtfulPictures_AskForTheirGroup()
    {
        // Confident suggestions are simply used: no category change is offered.
        var table = Job("table", Classified.As(ImageCategory.Table, 0.99));
        Assert.False(Wording.IsCategoryDoubtful(table));
        Assert.False(new ImageItemViewModel(table, 0, _root.Path).IsCategoryDoubtful);
        Assert.False(Wording.IsCategoryDoubtful(Job("read", Classified.As(ImageCategory.Chemical, 0.99, Classified.Read()))));

        var unsure = Job("unsure", Classified.As(ImageCategory.Other, 0.39));
        Assert.True(Wording.IsCategoryDoubtful(unsure));
        Assert.Equal("Not sure what kind of picture this is: it looks most like a photo or other image (39 % sure). Which kind of picture is it?",
            Wording.CategoryQuestion(unsure));

        var possible = Job("possible", Classified.As(ImageCategory.Chemical, 0.9, Classified.Uncertain()));
        Assert.True(Wording.IsCategoryDoubtful(possible));
        Assert.StartsWith("This looks like a chemical structure, but its atoms and bonds could not be read reliably.", Wording.CategoryQuestion(possible));

        Assert.True(Wording.IsCategoryDoubtful(Job("unclassified")));
        Assert.Contains("could not be sorted into a group", Wording.CategoryQuestion(Job("unclassified")));

        var sheet = Job("sheet", Classified.As(ImageCategory.Table, 0.7, null,
            new ClassifiedPanel(2, [0, 0, 10, 10], ImageCategory.Other, 0.45, "other", true, null)));
        Assert.True(Wording.IsCategoryDoubtful(sheet));
        Assert.StartsWith("Panel 2 of this sheet is hard to recognise.", Wording.CategoryQuestion(sheet));

        table.ManualCategory = ImageCategory.Chemical; // a choice the user made can always be changed back
        Assert.True(Wording.IsCategoryDoubtful(table));
        Assert.StartsWith("You set this picture's group to chemical diagram.", Wording.CategoryQuestion(table));
    }

    [Theory]
    [InlineData("already_good", null, "Already clear. Nothing needed fixing, so the original is kept exactly as it is.")]
    [InlineData("vector", "SVG is a vector graphic", "This is a drawing made of lines and shapes (a vector graphic)")]
    [InlineData("decode_failed", "cannot decode", "The picture's data is damaged and could not be opened.")]
    [InlineData("worker_crashed", "the image processor stopped", "The image processor stopped twice")]
    [InlineData("too_large", "processing would need about 9000 MB", "Too large to process safely on this computer, so the original is kept (processing would need about 9000 MB).")]
    [InlineData("something_new", "a new reason", "A new reason.")]
    public void KeptOrFailed_ExplainsInPlainWords(string reason, string? error, string expected)
    {
        var job = Job("j");
        job.Status = ProcessingStatus.Skipped;
        job.Reason = reason;
        job.Error = error;
        Assert.StartsWith(expected, Wording.KeptOrFailed(job));
        Assert.StartsWith(expected, new ImageItemViewModel(job, 0, _root.Path).ReviewSummary);
    }

    [Fact]
    public void NonChemicalPictures_HideChemicalErrorsAndRedrawDetails()
    {
        // A table whose redraw attempt left chemical findings behind (an older worker): nothing chemical is shown.
        var table = Job("table", Classified.As(ImageCategory.Table, 0.99));
        table.Redraw = new RedrawResult(false, ["not recognised"], null, null, null, null, [], 0.4, null,
            Findings: [new RedrawFinding("The AI is not sure about this atom (O, 53 % sure).", [1, 1, 5, 5])], IsChemical: true);
        var row = new ImageItemViewModel(table, 0, _root.Path);
        Assert.False(row.ShowsChemistry);
        Assert.False(row.HasMismatches);
        Assert.Empty(row.MismatchMarks);
        Assert.False(row.HasRedraw);
        Assert.False(row.HasChemicalStatus);
        Assert.False(row.IsPossibleChemical);
        Assert.DoesNotContain("53 % sure", ReportBuilder.Build(new JobState
            { JobId = "j", SourceType = SourceType.Image, SourceFileName = "t.png", Images = [table] }, _root.Path));

        // The same picture marked chemical by the user shows them.
        table.ManualCategory = ImageCategory.Chemical;
        table.ManualChemicalCheck = Classified.Uncertain();
        row = new ImageItemViewModel(table, 0, _root.Path);
        Assert.True(row.HasMismatches);
        Assert.StartsWith("Marked as a chemical diagram by you.", row.ChemicalStatusText);
        Assert.True(row.IsPossibleChemical);
    }

    [Fact]
    public void ManualCategory_IsSavedAndRestored()
    {
        var job = Job("photo", Classified.As(ImageCategory.Other, 0.9));
        job.ManualCategory = ImageCategory.Chemical;
        job.ManualChemicalCheck = Classified.Uncertain();
        var restored = JsonSerializer.Deserialize<ImageJob>(JsonSerializer.Serialize(job, JobStateStore.JsonOptions), JobStateStore.JsonOptions)!;
        Assert.Equal(ImageCategory.Chemical, restored.ManualCategory);
        Assert.Equal(ImageCategory.Other, restored.Classification!.Category);
        Assert.Equal("uncertain", restored.ManualChemicalCheck!.Status);
        Assert.True(restored.IsPossibleChemical);
        Assert.Contains("\"manualCategory\": \"CHEMICAL\"", JsonSerializer.Serialize(job, JobStateStore.JsonOptions));
    }

    // ---------------------------------------------------------------- with the worker and the classifier

    [Fact]
    public async Task ImageFiles_AreClassified_FromTheUnchangedFiles_AndTheUserCanChangeACategory()
    {
        Assert.SkipUnless(TestSamples.ClassifierInstalled(), "the picture classifier is not installed (worker/setup_ai.py)");
        var folder = Path.Combine(_root.Path, "kinds");
        var files = TestSamples.CategorySet(folder);
        var hashes = files.ToDictionary(f => f, f => SHA256.HashData(File.ReadAllBytes(f)));
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")));
        var outcome = await job.RunAsync(new JobRequest(files)).WaitAsync(TimeSpan.FromMinutes(5));
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);

        var byName = job.State!.Images.ToDictionary(i => Path.GetFileNameWithoutExtension(i.Location.SourcePath!));
        Assert.Equal(ImageCategory.Table, byName["table"].Category);
        Assert.Equal(ImageCategory.FlowchartOrGraph, byName["chart"].Category);
        Assert.Equal(ImageCategory.Other, byName["photo"].Category);
        Assert.All(byName.Values, i => Assert.Equal("DocumentFigureClassifier v2.5", i.Classification!.Model));
        Assert.Null(byName["table"].Classification!.ChemicalCheck); // MolScribe never reads a table
        Assert.Null(byName["photo"].Classification!.ChemicalCheck);
        if (byName.TryGetValue("structure", out var structure))
        {
            Assert.Equal(ImageCategory.Chemical, structure.Category);
            Assert.NotNull(structure.Classification!.ChemicalCheck); // the second check ran (read, uncertain or unavailable)
        }

        Assert.All(files, f => Assert.Equal(hashes[f], SHA256.HashData(File.ReadAllBytes(f)))); // the files are only read

        // The user marks the photo as a chemical diagram: MolScribe's check runs on it, and the choice is saved.
        var photo = byName["photo"];
        await job.SetCategoryAsync(photo.Id, ImageCategory.Chemical);
        var updated = job.SnapshotImages().Single(i => i.Id == photo.Id);
        Assert.Equal(ImageCategory.Chemical, updated.Category);
        Assert.NotNull(updated.ManualChemicalCheck);
        Assert.True(updated.IsPossibleChemical); // a photo cannot be read as a molecule: it needs review
        var saved = (await JobStateStore.LoadAsync(job.Workspace!))!;
        Assert.Equal(ImageCategory.Chemical, saved.Images.Single(i => i.Id == photo.Id).ManualCategory);

        await job.SetCategoryAsync(photo.Id, null); // back to the suggestion
        Assert.Equal(ImageCategory.Other, job.SnapshotImages().Single(i => i.Id == photo.Id).Category);
    }

    [Fact]
    public async Task DocxImages_AreClassifiedFromExtractedPartsCopiedByteForByte()
    {
        Assert.SkipUnless(TestSamples.ClassifierInstalled(), "the picture classifier is not installed (worker/setup_ai.py)");
        var images = Path.Combine(_root.Path, "set");
        Directory.CreateDirectory(images);
        TestSamples.ImageSet(images);
        var docx = Path.Combine(_root.Path, "report.docx");
        DocxSamples.Build(docx, images);
        await using var job = new JobOrchestrator(new WorkspaceManager(Path.Combine(_root.Path, "jobs")));
        var outcome = await job.RunAsync(new JobRequest([docx])).WaitAsync(TimeSpan.FromMinutes(5));
        Assert.Equal(JobOutcomeKind.ReadyForReview, outcome.Kind);

        using var zip = ZipFile.OpenRead(docx);
        var extracted = job.State!.Images.Where(i => i.ExtractedPath is not null).ToList();
        Assert.NotEmpty(extracted);
        foreach (var image in extracted)
        {
            using var part = zip.GetEntry(image.Location.ImageUri!.TrimStart('/'))!.Open();
            using var stored = new MemoryStream();
            part.CopyTo(stored);
            Assert.Equal(stored.ToArray(), File.ReadAllBytes(Path.Combine(job.Workspace!.RootPath, image.ExtractedPath!)));
            Assert.NotNull(image.Classification); // classified from that exact copy
        }
    }
}
