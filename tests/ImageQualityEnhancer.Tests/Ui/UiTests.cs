using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;
using ImageQualityEnhancer.ViewModels;
using ImageQualityEnhancer.Views;

namespace ImageQualityEnhancer.Tests.Ui;

internal sealed class FakeDialogs(params string[] files) : IFileDialogs
{
    public string[] Files { get; set; } = files;
    public string? SaveFolder { get; set; }
    public string? SuggestedName { get; private set; }
    public Task<IReadOnlyList<string>> PickInputFilesAsync() => Task.FromResult<IReadOnlyList<string>>(Files);

    public Task<string?> PickSaveFileAsync(string suggestedName, string extension)
    {
        SuggestedName = suggestedName;
        return Task.FromResult(SaveFolder is null ? null : Path.Combine(SaveFolder, suggestedName));
    }

    public Task<string?> PickFolderAsync() => Task.FromResult(SaveFolder);
}

/// <summary>
/// Drives the real views and view models off-screen: choose files, watch progress, review in the gallery,
/// approve, adjust manually, close and reopen. Screenshots are written when IQE_SCREENSHOT_DIR is set.
/// </summary>
public class UiTests : IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromMinutes(3);
    private readonly TempDirectory _root = new();
    private readonly string _images;

    public UiTests()
    {
        _images = Path.Combine(_root.Path, "images");
        Directory.CreateDirectory(_images);
        TestSamples.ImageSet(_images);
    }

    public void Dispose() => _root.Dispose();

    private static async Task WaitUntil(Func<bool> condition, string what)
    {
        var deadline = DateTime.UtcNow + Timeout;
        while (!condition())
        {
            Dispatcher.UIThread.RunJobs();
            if (DateTime.UtcNow > deadline)
            {
                throw new TimeoutException("Timed out waiting for: " + what);
            }

            await Task.Delay(50);
        }

        Dispatcher.UIThread.RunJobs();
    }

    private static void Screenshot(Window window, string name)
    {
        var dir = Environment.GetEnvironmentVariable("IQE_SCREENSHOT_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        Directory.CreateDirectory(dir);
        Dispatcher.UIThread.RunJobs();
        using var file = File.Create(Path.Combine(dir, name + ".png"));
        window.CaptureRenderedFrame()?.Save(file, new Avalonia.Media.Imaging.PngBitmapEncoderOptions());
    }

    private (Window Window, MainViewModel Vm) Open(params string[] files) => Open(new FakeDialogs(files));

    private (Window Window, MainViewModel Vm) Open(FakeDialogs dialogs)
    {
        var vm = new MainViewModel(new WorkspaceManager(Path.Combine(_root.Path, "jobs")), dialogs);
        Assert.False(vm.Start.AiRepair); // AI is optional: offered on the comparison screen after normal processing
        var window = new MainWindow { DataContext = vm, Width = 1280, Height = 820 };
        window.Show();
        return (window, vm);
    }

    private string[] ImageFiles() =>
        new[] { "photo.png", "photo.jpg", "scan.png", "small.gif" }.Select(n => Path.Combine(_images, n)).ToArray();

    [AvaloniaFact]
    public async Task StartPage_Shows()
    {
        var (window, vm) = Open();
        Assert.IsType<StartView>(window.GetVisualDescendants().OfType<StartView>().Single());
        Assert.IsType<StartViewModel>(vm.CurrentPage);
        Assert.True(vm.Start.IsImproveOnly); // improve only is the default
        vm.Start.SetStructuresCommand.Execute("redraw");
        Assert.True(vm.Start.RedrawStructures);
        Assert.False(vm.Start.IsImproveOnly);
        Assert.Contains("verified", vm.Start.StructuresDescription);
        Assert.True(vm.Start.IsRedrawProven);
        vm.Start.SetStructuresCommand.Execute("all"); // redraw every structure, unproven ones marked
        Assert.True(vm.Start.RedrawStructures && vm.Start.RedrawAll && vm.Start.IsRedrawAll);
        Assert.False(vm.Start.IsRedrawProven);
        Assert.Contains("not verified", vm.Start.StructuresDescription);
        vm.Start.SetStructuresCommand.Execute("improve");
        Assert.False(vm.Start.RedrawStructures);
        Assert.False(vm.Start.RedrawAll);
        Screenshot(window, "01-start");
        window.Close();
        await Task.CompletedTask;
    }

    [AvaloniaFact]
    public async Task ReviewFlow_ProgressGalleryApprovalAndManualAdjustment()
    {
        var (window, vm) = Open(ImageFiles());
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        Assert.Single(window.GetVisualDescendants().OfType<JobView>());

        await WaitUntil(() => job.IsReady, "processing to finish");
        Assert.Equal(100, job.Percent);
        Assert.NotNull(job.SelectedItem); // the first image to decide opens by itself
        Assert.Equal(4, job.Items.Count);
        Assert.All(job.Items, i => Assert.True(i.Job.Status == ProcessingStatus.Completed, $"{i.Title}: {i.Job.Status} {i.Job.Reason} {i.Job.Error}"));
        Assert.All(job.Items, i => Assert.Equal(i.Job.PixelSize, i.Job.OutputPixelSize)); // exact original pixel size

        // The gallery list is virtualised and shows rows with thumbnails.
        var list = window.GetVisualDescendants().OfType<ListBox>().Single();
        Assert.IsType<VirtualizingStackPanel>(list.ItemsPanelRoot);
        await WaitUntil(() => job.Items.All(i => i.OriginalThumbnail is not null && i.EnhancedThumbnail is not null), "thumbnails");

        // Select the diagram: it needs review and is excluded from "Approve all eligible".
        var scan = job.Items.Single(i => i.Title == "scan.png");
        Assert.True(scan.Job.NeedsReview);
        job.SelectedItem = scan;
        await WaitUntil(() => job.OriginalPreview is not null && job.EnhancedPreview is not null, "full previews");
        Screenshot(window, "02-review");

        // Levels: enhance the diagram again at level 4 with bolder text; both levels' accuracy can be compared.
        Assert.True(scan.IsLevel2);
        Assert.NotNull(scan.Job.TextCheck); // the diagram's caption was read before and after
        Assert.StartsWith("Text check:", scan.TextCheckText);
        job.SelectedBold = "On";
        await job.ApplyLevelCommand.ExecuteAsync("maximum");
        await WaitUntil(() => scan.IsLevel4, "level 4 result");
        Assert.Contains("bolden", scan.Job.Operations);
        Assert.Equal(scan.Job.PixelSize, scan.Job.OutputPixelSize);
        Assert.True(scan.Job.Settings.BoldText);
        Assert.StartsWith("Accuracy:", scan.AccuracyText);
        Assert.Contains("Level 4 · Maximum", scan.AccuracyText);
        Assert.Contains("2 Standard", job.LevelHistoryText);
        Assert.Contains("4 Maximum", job.LevelHistoryText);
        Screenshot(window, "02b-levels");

        await job.ApproveAllEligibleCommand.ExecuteAsync(null);
        Assert.False(scan.IsApproved);
        Assert.Contains("need review", job.Notice);
        Assert.Same(scan, job.SelectedItem);
        await job.ApproveCommand.ExecuteAsync(scan);
        await WaitUntil(() => scan.IsApproved, "approval shown"); // applied on the next UI tick
        // Deciding the image on screen moves on to the next image still to decide.
        Assert.NotSame(scan, job.SelectedItem);
        Assert.Equal(ApprovalStatus.Undecided, job.SelectedItem!.Job.Approval);

        // Rarely used actions sit in the "⋯" menu, which shares the job's data.
        var more = window.GetVisualDescendants().OfType<Button>().Single(b => b.Name == "MoreButton");
        var flyout = Assert.IsType<Flyout>(more.Flyout);
        flyout.ShowAt(more);
        Dispatcher.UIThread.RunJobs();
        Assert.Same(job, Assert.IsType<StackPanel>(flyout.Content).DataContext);
        Screenshot(window, "02c-more-menu");
        flyout.Hide();

        // Manual adjustment: enhance one image again with only sharpening.
        var photo = job.Items.Single(i => i.Title == "photo.png");
        job.SelectedItem = photo;
        Assert.True(job.CanAdjustSelection);
        job.ManualMode = true;
        job.DenoiseStrength = 0;
        job.ContrastStrength = 0;
        job.SharpenAmount = 1.0;
        await job.ReenhanceCommand.ExecuteAsync(null);
        await WaitUntil(() => photo.Job.Operations.SequenceEqual(["sharpen"]), "re-enhanced result");
        Assert.Equal(ApprovalStatus.Undecided, photo.Job.Approval); // a new result needs a new decision
        Screenshot(window, "03-manual");

        // Filters.
        job.SelectedFilter = job.FilterOptions.Single(f => f.Value == GalleryFilter.Approved);
        Assert.All(job.VisibleItems, i => Assert.True(i.IsApproved));
        job.SelectedFilter = job.FilterOptions[0];

        // Close and reopen from the start page: decisions are kept.
        await vm.BackToStartCommand.ExecuteAsync(null);
        var saved = Assert.Single(vm.Start.SavedJobs);
        Assert.Contains("ready for review", saved.Description);
        Assert.Contains("Open History", vm.Start.Message);
        Dispatcher.UIThread.RunJobs();
        var history = window.GetVisualDescendants().OfType<Border>().Single(b => b.Name == "HistoryPanel");
        Assert.False(history.IsVisible); // saved jobs are not on the start page itself
        Assert.Equal(1, vm.Start.SavedJobCount);
        vm.Start.ToggleHistoryCommand.Execute(null);
        Dispatcher.UIThread.RunJobs();
        Assert.True(history.IsVisible);
        Screenshot(window, "04-saved-jobs");
        vm.Start.ResumeJobCommand.Execute(saved);
        var reopened = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => reopened.IsReady, "reopen");
        Assert.True(reopened.Items.Single(i => i.Title == "scan.png").IsApproved);

        await vm.ShutdownAsync();
        window.Close();
    }

    [AvaloniaFact]
    public async Task RefusedRedraw_ShowsWhatDoesNotMatch_OnThePicture()
    {
        Assert.SkipUnless(TestSamples.RedrawInstalled(), "redraw is not installed for the worker's Python");
        var file = Path.Combine(_root.Path, "structure with caption.png");
        TestSamples.StructureWithCaption(file);
        var (window, vm) = Open(file);
        vm.Start.SetStructuresCommand.Execute("redraw");
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing");
        var item = Assert.Single(job.Items);
        job.SelectedItem = item;
        await WaitUntil(() => job.OriginalPreview is not null, "preview");

        Assert.False(item.Job.IsRedrawn);
        Assert.Contains("not part of the structure", item.MismatchText);
        Assert.StartsWith("Structure checks", item.MismatchSummary);
        Assert.Null(job.SelectedMarks);
        var checks = window.GetVisualDescendants().OfType<Expander>().Single(e => e.Name == "StructureChecks");
        Assert.False(checks.IsExpanded);
        Screenshot(window, "12-structure-checks-collapsed");
        checks.IsExpanded = true;
        job.ShowMismatches = true;
        Assert.NotEmpty(job.SelectedMarks!);
        var original = window.GetVisualDescendants().OfType<ZoomCompareView>().Single(v => v.Name == "OriginalView");
        Assert.Equal(job.SelectedMarks!, original.Marks!); // the red boxes are drawn on the original
        Screenshot(window, "12-structure-mismatches");
        checks.IsExpanded = false;
        Assert.False(job.ShowMismatches);
        Assert.Null(original.Marks);

        await vm.ShutdownAsync();
        window.Close();
    }

    [AvaloniaFact]
    public async Task RedrawAll_Table_ShowsNormalReviewWithoutChemicalErrors()
    {
        Assert.SkipUnless(TestSamples.RedrawInstalled(), "redraw is not installed for the worker's Python");
        var file = Path.Combine(_root.Path, "ordinary table.png");
        TestSamples.OrdinaryTable(file);
        var (window, vm) = Open(file);
        vm.Start.SetStructuresCommand.Execute("all");
        vm.Start.SetLevelCommand.Execute("maximum");
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing table");
        var item = Assert.Single(job.Items);
        job.SelectedItem = item;
        await WaitUntil(() => job.OriginalPreview is not null, "table preview");
        // A clean digital table has nothing to clean: kept as it is, or enhanced, but never treated as chemistry.
        Assert.Contains(item.Job.Status, new[] { ProcessingStatus.Completed, ProcessingStatus.Skipped });
        Assert.All(item.Job.Panels, panel => Assert.NotEqual(true, panel.Redraw?.IsChemical));
        Assert.DoesNotContain("redraw", item.Job.Operations);
        Assert.False(item.HasMismatches);
        Assert.False(item.HasMismatchSummary);
        Assert.Empty(item.MismatchMarks);
        Assert.DoesNotContain("chemist", item.Explanation ?? "");
        Assert.DoesNotContain("chemist", item.ReviewSummary ?? "");
        Assert.DoesNotContain("atom", item.ReviewSummary ?? "");
        Assert.DoesNotContain("not redrawn", item.RedrawText);
        var warning = window.GetVisualDescendants().OfType<Border>().Single(b => b.Classes.Contains("mismatch"));
        Assert.False(warning.IsVisible);
        Screenshot(window, "13-table-without-chemical-warnings");
        await vm.ShutdownAsync();
        window.Close();
    }

    [AvaloniaFact]
    public async Task ReviewScreen_PhotoHasNoChemicalDetails_AndMarksResetWhenSwitchingImages()
    {
        var workspaces = new WorkspaceManager(Path.Combine(_root.Path, "ui-jobs"));
        await using var job = new JobViewModel(new JobOrchestrator(workspaces), new FakeDialogs(), "Review images", (_, _) => { });
        var photo = new ImageJob
        {
            Id = "photo", DocumentId = "doc", ResourceId = "photo", ResourceRef = "photo", SourceType = SourceType.Image,
            Location = new ImageLocation { SourcePath = Path.Combine(_images, "photo.png") },
            Status = ProcessingStatus.Completed, ContentKind = "photo", NeedsReview = true,
            Operations = ["sharpen"], ReviewReasons = ["ai_restored"], PixelSize = new PixelSize(300, 200),
            Redraw = new RedrawResult(false, ["low confidence"], null, null, null, null, [], 0.4, null,
                Findings: [new RedrawFinding("The AI is not sure about this atom (Ag, 40 % sure).", [10, 10, 50, 50])]),
        };
        var structure = photo.Clone();
        structure.ContentKind = "text_or_diagram";
        structure.Redraw = photo.Redraw with { IsChemical = true };
        var photoRow = new ImageItemViewModel(photo, 0, _root.Path);
        var structureRow = new ImageItemViewModel(structure, 1, _root.Path);
        job.IsReady = true;
        job.StageText = "Ready for review";
        job.Items.Add(photoRow);
        job.VisibleItems.Add(photoRow);
        job.SelectedItem = structureRow;
        job.StructureChecksExpanded = true;
        job.ShowMismatches = true;
        Assert.NotEmpty(job.SelectedMarks!);
        job.SelectedItem = photoRow;
        Assert.False(job.StructureChecksExpanded);
        Assert.False(job.ShowMismatches);
        Assert.Null(job.SelectedMarks);
        var window = new Window { Content = new JobView { DataContext = job }, Width = 1280, Height = 820 };
        window.Show();
        await WaitUntil(() => job.OriginalPreview is not null, "photo preview");
        var warning = window.GetVisualDescendants().OfType<Border>().Single(b => b.Classes.Contains("mismatch"));
        Assert.False(warning.IsVisible);
        Assert.Contains("Compare the photo", photoRow.ReviewSummary);
        Assert.All(window.GetVisualDescendants().OfType<Expander>(), e => Assert.False(e.IsExpanded));
        Screenshot(window, "14-photo-simple-review");
        window.Close();
    }

    [AvaloniaFact]
    public async Task Groups_CountAndFilter_ManualCategoryChange_AndChemistryHiddenForTables()
    {
        Assert.SkipUnless(TestSamples.ClassifierInstalled(), "the picture classifier is not installed (worker/setup_ai.py)");
        var files = TestSamples.CategorySet(Path.Combine(_root.Path, "kinds")).ToArray();
        var hasStructure = files.Any(f => f.EndsWith("structure.png"));
        var (window, vm) = Open(files);
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing");

        string Group(ImageCategory? category) => job.CategoryFilters.Single(f => f.Category == category).Label;
        Assert.Equal($"All  {files.Length}", Group(null));
        Assert.Equal($"Chemical diagrams  {(hasStructure ? 1 : 0)}", Group(ImageCategory.Chemical));
        Assert.Equal("Flowcharts and graphs  1", Group(ImageCategory.FlowchartOrGraph));
        Assert.Equal("Tables  1", Group(ImageCategory.Table));
        Assert.Equal("Other images  2", Group(ImageCategory.Other)); // the photo and the unsure scan
        var bar = window.GetVisualDescendants().OfType<ItemsControl>().Single(c => c.Name == "CategoryBar");
        Assert.Equal(5, bar.ItemCount);

        // Filter: tables only. A table shows its suggestion and confidence, and nothing chemical.
        job.SelectCategoryCommand.Execute(nameof(ImageCategory.Table));
        Dispatcher.UIThread.RunJobs();
        var table = Assert.Single(job.VisibleItems);
        Assert.EndsWith("table.png", table.Job.Location.SourcePath);
        job.SelectedItem = table;
        await WaitUntil(() => job.OriginalPreview is not null, "preview");
        Assert.Equal(ImageCategory.Table, job.SelectedCategoryChoice!.Category);
        Assert.StartsWith("Suggested by DocumentFigureClassifier v2.5: Table", table.ClassificationDetails);
        Assert.False(table.HasChemicalStatus);
        Assert.False(table.HasMismatches);
        var chemistry = window.GetVisualDescendants().OfType<TextBlock>().Single(t => t.Name == "ChemicalStatus");
        var question = window.GetVisualDescendants().OfType<Border>().Single(b => b.Name == "CategoryQuestion");
        Assert.False(chemistry.IsEffectivelyVisible);
        Assert.False(question.IsEffectivelyVisible); // a confident suggestion is not offered for change
        Screenshot(window, "14-groups-tables");

        // The classifier is unsure about the faded scan: it asks, and the user marks it as a chemical diagram.
        job.SelectCategoryCommand.Execute("All");
        var unsure = job.VisibleItems.Single(i => i.Job.Location.SourcePath!.EndsWith("unsure.png"));
        job.SelectedItem = unsure;
        Dispatcher.UIThread.RunJobs();
        Assert.True(unsure.IsCategoryDoubtful);
        Assert.True(question.IsEffectivelyVisible);
        Assert.Contains("Which kind of picture is it?", unsure.CategoryQuestion);
        job.SelectedCategoryChoice = job.CategoryChoices.Single(c => c.Category == ImageCategory.Chemical);
        await WaitUntil(() => !job.IsBusy && unsure.Job.ManualCategory == ImageCategory.Chemical, "category change");
        Assert.Equal($"Chemical diagrams  {(hasStructure ? 2 : 1)}", Group(ImageCategory.Chemical));
        Assert.Equal("Other images  1", Group(ImageCategory.Other));
        Assert.Equal("Chemical diagram · set by you", unsure.CategoryLine);
        Assert.StartsWith("Marked as a chemical diagram by you.", unsure.ChemicalStatusText);
        Assert.DoesNotContain("verified by the automatic checks", unsure.ChemicalStatusText);
        Assert.True(chemistry.IsEffectivelyVisible);
        Assert.True(question.IsEffectivelyVisible); // the user's own choice can be changed back
        Screenshot(window, "15-groups-marked-chemical");

        await job.UseSuggestedCategoryCommand.ExecuteAsync(null);
        await WaitUntil(() => !job.IsBusy && unsure.Job.ManualCategory is null, "back to the suggestion");
        Assert.Equal("Other images  2", Group(ImageCategory.Other));

        await vm.ShutdownAsync();
        window.Close();
    }

    [AvaloniaFact]
    public async Task OptionalAi_SuggestsFirst_TriesOnOneOrAll_AndAsksBeforeChangingAnApproval()
    {
        Assert.SkipUnless(TestSamples.AiRepairInstalled(), "the AI repair models are not installed (worker/setup_ai.py)");
        var table = Path.Combine(_root.Path, "table.png");
        TestSamples.OrdinaryTable(table);
        var (window, vm) = Open(Path.Combine(_images, "photo.png"), Path.Combine(_images, "small.gif"), Path.Combine(_images, "photo.jpg"), table);
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing to finish");
        Assert.All(job.Items, i => Assert.DoesNotContain(i.Job.Operations, o => o.StartsWith("ai_", StringComparison.Ordinal))); // no AI until asked

        Button Named(string name) => window.GetVisualDescendants().OfType<Button>().Single(b => b.Name == name);
        T Find<T>(string name) where T : Control => window.GetVisualDescendants().OfType<T>().Single(c => c.Name == name);

        // A clean table: no AI offered.
        job.SelectedItem = job.Items.Single(i => i.Title == "table.png");
        await WaitUntil(() => job.OriginalPreview is not null, "table preview");
        Assert.Equal("No suitable AI repair detected.", Find<TextBlock>("AiText").Text);
        Assert.False(Named("TryAiButton").IsVisible);

        // The grainy GIF, still to decide: a suggestion that promises nothing, and the button.
        var gif = job.Items.Single(i => i.Title == "small.gif");
        job.SelectedItem = gif;
        await WaitUntil(() => job.OriginalPreview is not null && job.EnhancedPreview is not null, "GIF previews");
        Assert.Equal("Grain detected — SCUNet may help.", Find<TextBlock>("AiText").Text);
        Assert.True(Named("TryAiButton").IsVisible && Named("TryAiButton").IsEnabled);
        Assert.False(Find<StackPanel>("AiHeadline").IsVisible);
        Assert.Equal(3, job.EligibleAiCount);
        Assert.Equal("Try AI on eligible images (3)", Named("TryAiOnEligibleButton").Content);
        Assert.False(Find<StackPanel>("ComparePairs").IsVisible);
        Screenshot(window, "16-ai-suggestion");

        // Click "Try AI enhancement": the screen says clearly what happened.
        await job.TryAiCommand.ExecuteAsync(null);
        await WaitUntil(() => gif.Job.Ai is not null, "the AI result");
        Assert.True(Find<StackPanel>("AiHeadline").IsVisible);
        Assert.False(Named("TryAiButton").IsVisible); // tried once on this result
        Assert.Equal(2, job.EligibleAiCount);
        Assert.Equal(ApprovalStatus.Undecided, gif.Job.Approval); // never approved for the user
        if (gif.IsAiAccepted)
        {
            // Applied: the AI version is in use, marked in the panel, the notice, the picture label, the badge and the row.
            Assert.True(gif.Job.UseAiVersion);
            Assert.Equal(gif.Job.Ai!.Path, gif.Job.ResultPath);
            Assert.Equal("AI enhancement applied (SCUNet)", Find<TextBlock>("AiHeadlineText").Text);
            Assert.Equal("✓", gif.AiHeadlineIcon);
            Assert.Contains("aiApplied", Find<Border>("AiPanel").Classes);
            Assert.StartsWith("The AI version is now used for this picture.", Find<TextBlock>("AiText").Text);
            Assert.StartsWith("✓ AI enhancement applied to small.gif (SCUNet).", job.Notice);
            Assert.Equal("opencv-ai", job.ComparePair); // what AI changed, side by side
            Assert.Equal("OpenCV result", job.LeftLabel);
            Assert.Equal("AI version (SCUNet) · in use", job.RightLabel);
            Assert.Equal("AI enhanced · needs review", gif.StatusText);
            Assert.Equal("✓ AI enhancement applied (SCUNet)", gif.AiRowText);
            Assert.True(Find<StackPanel>("ComparePairs").IsVisible);
            var viewers = window.GetVisualDescendants().OfType<ZoomCompareView>().Where(v => v.IsEffectivelyVisible).ToList();
            await WaitUntil(() => viewers.Count == 2 && viewers.All(v => v.Source is not null), "OpenCV and AI pictures on screen");
            Screenshot(window, "17-ai-applied");

            // Switch back: undecided, so no question; the AI version stays available.
            await job.UseVersionCommand.ExecuteAsync("opencv");
            Assert.False(job.HasPendingVersion);
            await WaitUntil(() => !gif.Job.UseAiVersion, "the OpenCV result to be used again");
            Assert.Equal("AI version ready (SCUNet) · not in use", Find<TextBlock>("AiHeadlineText").Text);
            Assert.DoesNotContain("aiApplied", Find<Border>("AiPanel").Classes);
            Assert.Equal("original-opencv", job.ComparePair); // the version in use, against the original
            Assert.Equal("OpenCV result · in use", job.RightLabel);
            job.SetComparePairCommand.Execute("opencv-ai");
            Assert.Equal("OpenCV result · in use", job.LeftLabel);
            Assert.Equal("AI version (SCUNet) · not in use", job.RightLabel);
            await job.UseVersionCommand.ExecuteAsync("ai");
            await WaitUntil(() => gif.Job.UseAiVersion, "the AI version to be used again");
        }
        else
        {
            Assert.False(gif.Job.UseAiVersion);
            Assert.Equal("AI enhancement not applied (SCUNet)", Find<TextBlock>("AiHeadlineText").Text);
            Assert.StartsWith("It did not pass the checks:", Find<TextBlock>("AiText").Text);
            Assert.EndsWith("The OpenCV result stays in use.", Find<TextBlock>("AiText").Text);
            Screenshot(window, "17-ai-not-applied");
        }

        // An approved picture: "Try AI enhancement" keeps the approval and asks at once whether to use the AI version.
        var photo = job.Items.Single(i => i.Title == "photo.png");
        await job.ApproveCommand.ExecuteAsync(photo);
        await WaitUntil(() => photo.Job.Approval == ApprovalStatus.Approved, "the approval");
        job.SelectedItem = photo; // approving moved on to the next picture to decide
        await job.TryAiCommand.ExecuteAsync(null);
        await WaitUntil(() => photo.Job.Ai is not null, "the AI result for photo.png");
        Assert.Equal(ApprovalStatus.Approved, photo.Job.Approval);
        Assert.False(photo.Job.UseAiVersion); // not switched without asking
        if (photo.IsAiAccepted)
        {
            Assert.True(job.HasPendingVersion); // the question is on screen
            Assert.Equal("AI version ready (SCUNet) · not in use", Find<TextBlock>("AiHeadlineText").Text);
            Assert.Contains("You had already approved the OpenCV result", Find<TextBlock>("AiText").Text);
            Assert.Equal("photo.png: AI version ready (SCUNet) · not in use. It passed the detail check.", job.Notice);
            Assert.Contains("It will then need your approval again", job.PendingVersionText);
            Screenshot(window, "18-ai-confirm-switch");

            job.CancelVersionSwitchCommand.Execute(null); // "No, keep my approval"
            Assert.False(photo.Job.UseAiVersion);
            Assert.Equal(ApprovalStatus.Approved, photo.Job.Approval);

            await job.UseVersionCommand.ExecuteAsync("ai");
            Assert.True(job.HasPendingVersion);
            await job.ConfirmVersionSwitchCommand.ExecuteAsync(null); // "Yes, switch"
            await WaitUntil(() => photo.Job.UseAiVersion, "the AI version to be used");
            Assert.Equal(ApprovalStatus.Undecided, photo.Job.Approval);
            Assert.Equal("AI enhancement applied (SCUNet)", Find<TextBlock>("AiHeadlineText").Text);
            Screenshot(window, "18b-ai-version-in-use");
        }

        // "Try AI on eligible images": only the remaining eligible picture is tried, with progress; nothing is approved.
        var approvals = job.Items.ToDictionary(i => i.Id, i => i.Job.Approval);
        var jpeg = job.Items.Single(i => i.Title == "photo.jpg");
        var run = job.TryAiOnEligibleCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();
        if (!run.IsCompleted)
        {
            Assert.True(job.IsAiRunning);
            Assert.False(Named("TryAiOnEligibleButton").IsVisible); // progress and Stop take its place
            Assert.Contains(window.GetVisualDescendants().OfType<ProgressBar>(), b => b.IsEffectivelyVisible && b.Height == 6);
            Assert.StartsWith("Trying AI", job.AiProgressText);
            Screenshot(window, "19-ai-progress");
        }

        await run.WaitAsync(Timeout);
        await WaitUntil(() => jpeg.Job.Ai is not null, "the AI result for photo.jpg");
        Assert.False(job.IsAiRunning);
        Assert.StartsWith($"AI tried on 1 picture: AI enhancement applied to {(jpeg.IsAiAccepted ? 1 : 0)} picture", job.Notice);
        Assert.Contains("Nothing was approved", job.Notice);
        Assert.Equal(jpeg.IsAiAccepted, jpeg.Job.UseAiVersion); // still to decide: applied when usable
        Assert.Null(job.Items.Single(i => i.Title == "table.png").Job.Ai); // ineligible: unchanged
        Assert.All(job.Items, i => Assert.Equal(approvals[i.Id], i.Job.Approval));
        Assert.Equal(0, job.EligibleAiCount);
        Assert.False(Named("TryAiOnEligibleButton").IsVisible); // nothing left to try
        job.SelectedItem = jpeg;
        await WaitUntil(() => job.OriginalPreview is not null, "photo.jpg preview");
        Screenshot(window, "20-ai-whole-job");
        window.Close();
    }

    [AvaloniaFact]
    public async Task PauseAndCancel_FromTheProgressView()
    {
        var copies = Enumerable.Range(1, 8).Select(i =>
        {
            var path = Path.Combine(_root.Path, $"p{i}.png");
            File.Copy(Path.Combine(_images, "photo.png"), path);
            return path;
        }).ToArray();
        var (window, vm) = Open(copies);
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);

        await WaitUntil(() => job.Items.Any(i => i.Job.Status == ProcessingStatus.Completed), "first image");
        job.PauseCommand.Execute(null);
        await WaitUntil(() => job.IsPaused, "paused");
        Assert.True(job.CanResume);
        Assert.False(job.CanPause);
        Screenshot(window, "05-paused");

        await job.CancelCommand.ExecuteAsync(null);
        await WaitUntil(() => vm.CurrentPage is StartViewModel, "back to start");
        Assert.Contains("cancelled", vm.Start.Message);
        Assert.Empty(vm.Start.SavedJobs);
        Assert.All(copies, f => Assert.True(File.Exists(f)));
        window.Close();
    }

    [AvaloniaFact]
    public async Task PasswordProtectedPdf_AsksForPassword_ThenContinues()
    {
        var pdf = Path.Combine(_root.Path, "secret.pdf");
        TestSamples.FeaturePdf(pdf, "s3cret");
        var (window, vm) = Open(pdf);
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);

        await WaitUntil(() => vm.CurrentPage is StartViewModel && vm.Start.NeedsPassword, "password prompt");
        Assert.Empty(vm.Start.SavedJobs); // shown once, as the password box
        Screenshot(window, "06-password");
        await vm.Start.DismissPasswordCommand.ExecuteAsync(null); // "Not now": the job waits in the list
        Assert.False(vm.Start.NeedsPassword);
        vm.Start.ResumeJobCommand.Execute(Assert.Single(vm.Start.SavedJobs));
        Assert.True(vm.Start.NeedsPassword);
        vm.Start.Password = "wrong";
        vm.Start.SubmitPasswordCommand.Execute(null);
        await WaitUntil(() => vm.CurrentPage is StartViewModel && vm.Start.NeedsPassword, "second prompt");
        Assert.Contains("not correct", vm.Start.PasswordPrompt);

        vm.Start.Password = "s3cret";
        vm.Start.SubmitPasswordCommand.Execute(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing with password");
        Assert.NotEmpty(job.Items);

        await vm.ShutdownAsync();
        window.Close();
    }

    [AvaloniaFact]
    public async Task Export_SignedPdf_RequiresConfirmation_ThenSavesAndFinishes()
    {
        var pdf = Path.Combine(_root.Path, "Signed report.pdf");
        TestSamples.SignedPdf(pdf);
        var outDir = Path.Combine(_root.Path, "exported");
        Directory.CreateDirectory(outDir);
        var dialogs = new FakeDialogs(pdf) { SaveFolder = outDir };
        var (window, vm) = Open(dialogs);
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing");
        Assert.True(job.IsSigned);
        Assert.False(job.CanExport); // nothing approved yet

        foreach (var item in job.Items.Where(i => i.CanApprove).ToList())
        {
            await job.ApproveCommand.ExecuteAsync(item);
        }

        await WaitUntil(() => job.CanExport, "approvals");
        await job.ExportCommand.ExecuteAsync(null);
        Assert.True(job.ShowSignedWarning); // warned, nothing written yet
        Assert.Empty(Directory.GetFiles(outDir));
        Screenshot(window, "07-signed-warning");

        await job.ConfirmSignedExportCommand.ExecuteAsync(null);
        await WaitUntil(() => job.ExportMessage is not null || job.ExportError is not null, "export");
        Assert.Null(job.ExportError);
        Assert.Equal("Signed report_enhanced.pdf", dialogs.SuggestedName);
        Assert.True(File.Exists(Path.Combine(outDir, "Signed report_enhanced.pdf")));
        Assert.Contains("Saved", job.ExportMessage);
        Assert.Contains("Images:", job.DiagnosticsText);
        Screenshot(window, "08-exported");

        await job.FinishCommand.ExecuteAsync(null);
        await WaitUntil(() => vm.CurrentPage is StartViewModel, "back to start");
        Assert.Empty(vm.Start.SavedJobs); // temporary files removed
        Assert.True(File.Exists(pdf));
        window.Close();
    }

    [AvaloniaFact]
    public async Task ZoomViewer_ZoomsInSync_AndSliderCompares()
    {
        var (window, vm) = Open(ImageFiles());
        await vm.Start.ChooseFilesCommand.ExecuteAsync(null);
        var job = Assert.IsType<JobViewModel>(vm.CurrentPage);
        await WaitUntil(() => job.IsReady, "processing");
        job.SelectedItem = job.Items.Single(i => i.Title == "scan.png");
        await WaitUntil(() => job.OriginalPreview is not null && job.EnhancedPreview is not null, "previews");

        var viewers = window.GetVisualDescendants().OfType<ZoomCompareView>().ToList();
        var original = viewers.Single(v => v.Name == "OriginalView");
        var enhanced = viewers.Single(v => v.Name == "EnhancedView");
        Assert.Equal("Fit", job.ZoomText);

        enhanced.ZoomAt(3.0, new Avalonia.Point(enhanced.Bounds.Width / 2, enhanced.Bounds.Height / 2));
        Dispatcher.UIThread.RunJobs();
        Assert.True(job.ViewZoom > 0);
        Assert.Equal(job.ViewZoom, original.Zoom);                       // side-by-side views stay in sync
        Assert.EndsWith("%", job.ZoomText);
        Screenshot(window, "09-zoomed-side-by-side");

        job.SetViewModeCommand.Execute("slider");
        job.SplitPosition = 0.45;
        Dispatcher.UIThread.RunJobs();
        Assert.True(viewers.Single(v => v.Name == "SliderView").IsEffectivelyVisible);
        Screenshot(window, "10-slider");

        job.ToggleExpandedCommand.Execute(null);
        job.ZoomFitCommand.Execute(null);
        Dispatcher.UIThread.RunJobs();
        Assert.Equal("Fit", job.ZoomText);
        Screenshot(window, "11-expanded");

        await vm.ShutdownAsync();
        window.Close();
    }
}
