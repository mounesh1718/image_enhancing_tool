using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.ViewModels;

public enum GalleryFilter
{
    All,
    ToDecide,
    NeedsReview,
    Approved,
    KeptOrSkipped,
    Failed,
}

public sealed record FilterOption(GalleryFilter Value, string Label)
{
    public override string ToString() => Label;
}

/// <summary>One group button above the image list: "Tables 61". <see cref="Category"/> null = all images.</summary>
public sealed record CategoryFilterOption(ImageCategory? Category, string Name, int Count, bool IsSelected)
{
    public string Label => $"{Name}  {Count:N0}";
    public string Key => Category?.ToString() ?? "All";

    /// <summary>An empty group is shown dimmed (nothing to list), unless it is the one selected.</summary>
    public bool IsAvailable => Category is null || Count > 0 || IsSelected;
}

/// <summary>A category the user can give the selected picture.</summary>
public sealed record CategoryChoice(ImageCategory Category, string Label)
{
    public override string ToString() => Label;
}

/// <summary>
/// Progress, pause/resume/cancel, the review gallery and manual adjustments for one job.
/// All orchestrator events arrive on background threads and are marshalled with the UI dispatcher.
/// </summary>
public sealed partial class JobViewModel : ViewModelBase, IAsyncDisposable
{
    private readonly JobOrchestrator _orchestrator;
    private readonly Action<JobOutcome, JobViewModel> _finished;
    private readonly CancellationTokenSource _appClosing = new();
    private readonly Dictionary<string, ImageItemViewModel> _items = [];
    private Task<JobOutcome>? _run;

    private readonly IFileDialogs _dialogs;

    public JobViewModel(JobOrchestrator orchestrator, IFileDialogs dialogs, string documentName, Action<JobOutcome, JobViewModel> finished)
    {
        _orchestrator = orchestrator;
        _dialogs = dialogs;
        _finished = finished;
        DocumentName = documentName;
        SelectedFilter = FilterOptions[0];
        UpdateCategoryFilters();
        _orchestrator.ProgressChanged += p => Dispatcher.UIThread.Post(() => ApplyProgress(p));
        _orchestrator.ImageUpdated += job => Dispatcher.UIThread.Post(() => ApplyImage(job));
    }

    public ObservableCollection<ImageItemViewModel> Items { get; } = [];
    public ObservableCollection<ImageItemViewModel> VisibleItems { get; } = [];
    public IReadOnlyList<FilterOption> FilterOptions { get; } =
    [
        new(GalleryFilter.All, "All images"),
        new(GalleryFilter.ToDecide, "To decide"),
        new(GalleryFilter.NeedsReview, "Needs a careful look"),
        new(GalleryFilter.Approved, "Approved"),
        new(GalleryFilter.KeptOrSkipped, "Kept original"),
        new(GalleryFilter.Failed, "Failed"),
    ];

    // ------------------------------------------------------------------ progress

    [ObservableProperty] public partial string DocumentName { get; set; }
    [ObservableProperty] public partial string StageText { get; set; } = "Starting…";
    [ObservableProperty] public partial string CurrentOperation { get; set; } = "";
    [ObservableProperty] public partial double Percent { get; set; }
    [ObservableProperty] public partial string CountsText { get; set; } = "";
    [ObservableProperty] public partial string? EtaText { get; set; }
    [ObservableProperty] public partial string? PauseReason { get; set; }
    [ObservableProperty] public partial string? ErrorText { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPause), nameof(CanResume), nameof(CanCancel))]
    public partial bool IsRunning { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPause), nameof(CanResume))]
    public partial bool IsPaused { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPause), nameof(CanResume), nameof(CanCancel), nameof(CanReview))]
    public partial bool IsReady { get; set; }

    public bool CanPause => IsRunning && !IsPaused;
    public bool CanResume => IsRunning && IsPaused;
    public bool CanCancel => IsRunning || IsReady;
    public bool CanReview => IsReady && !IsBusy;

    public JobWorkspace? Workspace => _orchestrator.Workspace;
    public string? WorkspaceRoot => Workspace?.RootPath;

    // ------------------------------------------------------------------ run

    public void Start(JobRequest request) => Track(_orchestrator.RunAsync(request, _appClosing.Token));

    public void Resume(JobWorkspace workspace, string? password) => Track(_orchestrator.ResumeAsync(workspace, password, _appClosing.Token));

    private void Track(Task<JobOutcome> run)
    {
        IsRunning = true;
        _run = run;
        run.ContinueWith(t => Dispatcher.UIThread.Post(() => OnFinished(t.IsCompletedSuccessfully
            ? t.Result
            : new JobOutcome(JobOutcomeKind.Failed, "internal_error", t.Exception?.GetBaseException().Message))), TaskScheduler.Default);
    }

    private void OnFinished(JobOutcome outcome)
    {
        IsRunning = false;
        IsPaused = false;
        if (outcome.Kind == JobOutcomeKind.Cancelled && _cancelling)
        {
            return; // CancelAsync reports the outcome once cleanup has finished
        }

        switch (outcome.Kind)
        {
            case JobOutcomeKind.ReadyForReview:
                IsReady = true;
                StageText = "Ready for review";
                CurrentOperation = "Check the results, then approve the enhancements you want to keep.";
                Percent = 100;
                EtaText = null;
                foreach (var job in _orchestrator.SnapshotImages())
                {
                    ApplyImage(job); // catch up on any update still queued
                }

                IsSigned = _orchestrator.State?.DigitallySigned == true;
                if (_orchestrator.State?.ClassifierUnavailable is { } why)
                {
                    Notice = "The images were not sorted into groups: " + why + ". Chemical redraw needs the classification.";
                }

                RefreshDiagnostics();
                // Open the first image that needs a decision, so review can start right away.
                SelectedItem ??= VisibleItems.FirstOrDefault(IsUndecided) ?? VisibleItems.FirstOrDefault();
                break;
            case JobOutcomeKind.Failed:
                StageText = "Stopped";
                ErrorText = outcome.Message;
                break;
        }

        _finished(outcome, this);
    }

    [RelayCommand]
    private void Pause() => _orchestrator.Pause();

    [RelayCommand]
    private void ResumeProcessing() => _orchestrator.Resume();

    [RelayCommand]
    private async Task CancelAsync()
    {
        _cancelling = true;
        StageText = "Cancelling…";
        CurrentOperation = "Stopping and removing temporary files";
        await _orchestrator.CancelAsync();
        // Reported only now, after the temporary files are gone, so the start page never lists this job.
        IsRunning = false;
        IsReady = false;
        _finished(new JobOutcome(JobOutcomeKind.Cancelled), this);
    }

    private bool _cancelling;

    /// <summary>Stops processing because the app is closing. Progress is saved so the job can be resumed later.</summary>
    public async Task StopForShutdownAsync()
    {
        _appClosing.Cancel();
        if (_run is not null)
        {
            await _run.ContinueWith(_ => { }, TaskScheduler.Default);
        }
    }

    private void ApplyProgress(JobProgress p)
    {
        if (IsExporting)
        {
            ExportProgress = p.Percent;
            ExportOperation = p.CurrentOperation;
            return;
        }

        if (IsReady)
        {
            return;
        }

        DocumentName = p.DocumentName;
        StageText = p.Stage switch
        {
            JobStage.Created => "Preparing",
            JobStage.Scanning => "Scanning the document",
            JobStage.Extracting => "Extracting images",
            JobStage.Classifying => "Classifying images",
            JobStage.Enhancing => "Enhancing images",
            JobStage.AwaitingReview => "Ready for review",
            JobStage.Failed => "Stopped",
            _ => p.Stage.ToString(),
        };
        CurrentOperation = p.CurrentOperation;
        Percent = p.Percent;
        CountsText = p.TotalImages == 0
            ? ""
            : $"{p.TotalImages:N0} images · {p.Processed:N0} processed · {p.Pending:N0} pending" +
              (p.Failed > 0 ? $" · {p.Failed:N0} failed" : "");
        EtaText = p.EstimatedRemaining is { } eta && eta > TimeSpan.Zero ? $"About {FormatDuration(eta)} remaining" : null;
        IsPaused = p.IsPaused;
        PauseReason = p.PauseReason;
    }

    private static string FormatDuration(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours} h {t.Minutes} min" : t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes} min" : "less than a minute";

    // ------------------------------------------------------------------ gallery

    [ObservableProperty] public partial FilterOption SelectedFilter { get; set; }

    partial void OnSelectedFilterChanged(FilterOption value) => RebuildVisible();

    private GalleryFilter Filter => SelectedFilter?.Value ?? GalleryFilter.All;

    [ObservableProperty] public partial string DecisionSummary { get; set; } = "";

    private void ApplyImage(ImageJob job)
    {
        if (WorkspaceRoot is null)
        {
            return;
        }

        if (_items.TryGetValue(job.Id, out var item))
        {
            item.Update(job);
        }
        else
        {
            item = new ImageItemViewModel(job, _items.Count, WorkspaceRoot);
            _items[job.Id] = item;
            Items.Add(item);
        }

        // Keep the filtered list in step without rebuilding it.
        var visible = VisibleItems.Contains(item);
        var matches = Matches(item);
        if (matches && !visible)
        {
            var at = 0;
            while (at < VisibleItems.Count && VisibleItems[at].Index < item.Index)
            {
                at++;
            }

            VisibleItems.Insert(at, item);
        }
        else if (!matches && visible && item != SelectedItem)
        {
            VisibleItems.Remove(item);
        }

        if (item == SelectedItem)
        {
            RefreshSelection();
        }

        UpdateSummary();
        UpdateCategoryFilters();
    }

    // ------------------------------------------------------------------ groups (chemical diagrams, tables, ...)

    /// <summary>The group buttons with their image counts. A sheet with panels of several kinds counts in each.</summary>
    public ObservableCollection<CategoryFilterOption> CategoryFilters { get; } = [];

    /// <summary>The group shown in the list; null shows every image.</summary>
    [ObservableProperty] public partial ImageCategory? SelectedCategory { get; set; }

    partial void OnSelectedCategoryChanged(ImageCategory? value)
    {
        RebuildVisible();
        UpdateCategoryFilters();
    }

    [RelayCommand]
    private void SelectCategory(string key) =>
        SelectedCategory = Enum.TryParse<ImageCategory>(key, out var category) ? category : null;

    // In this order the buttons fit two rows of the image list.
    private static readonly (ImageCategory Category, string Name)[] Groups =
    [
        (ImageCategory.Chemical, "Chemical diagrams"),
        (ImageCategory.Table, "Tables"),
        (ImageCategory.FlowchartOrGraph, "Flowcharts and graphs"),
        (ImageCategory.Other, "Other images"),
    ];

    private bool InSelectedGroup(ImageItemViewModel item) =>
        SelectedCategory is not { } category || item.Job.Groups.Contains(category);

    private void UpdateCategoryFilters()
    {
        var options = new List<CategoryFilterOption> { new(null, "All", Items.Count, SelectedCategory is null) };
        options.AddRange(Groups.Select(g => new CategoryFilterOption(g.Category, g.Name,
            Items.Count(i => i.Job.Groups.Contains(g.Category)), SelectedCategory == g.Category)));
        for (var k = 0; k < options.Count; k++)
        {
            if (k >= CategoryFilters.Count)
            {
                CategoryFilters.Add(options[k]);
            }
            else if (CategoryFilters[k] != options[k])
            {
                CategoryFilters[k] = options[k];
            }
        }
    }

    /// <summary>The categories the user can give a picture (any choice replaces the suggestion for all its panels).</summary>
    public IReadOnlyList<CategoryChoice> CategoryChoices { get; } =
    [
        new(ImageCategory.Chemical, "Chemical diagram"),
        new(ImageCategory.FlowchartOrGraph, "Flowchart or graph"),
        new(ImageCategory.Table, "Table"),
        new(ImageCategory.Other, "Other image (photo, logo, ...)"),
    ];

    /// <summary>The selected picture's category in the chooser; changing it records the user's choice.</summary>
    [ObservableProperty] public partial CategoryChoice? SelectedCategoryChoice { get; set; }

    private bool _syncingCategory;

    partial void OnSelectedCategoryChoiceChanged(CategoryChoice? value)
    {
        if (_syncingCategory || value is null || SelectedItem is not { } item || value.Category == item.Job.Category)
        {
            return;
        }

        _ = ChangeCategoryAsync(item, value.Category);
    }

    /// <summary>Back to the classifier's suggestion.</summary>
    [RelayCommand]
    private Task UseSuggestedCategory() => SelectedItem is { } item ? ChangeCategoryAsync(item, null) : Task.CompletedTask;

    private async Task ChangeCategoryAsync(ImageItemViewModel item, ImageCategory? category)
    {
        IsBusy = true;
        Notice = null;
        try
        {
            await _orchestrator.SetCategoryAsync(item.Id, category);
            Notice = category is null
                ? "The suggested category is used again."
                : $"Marked as {Wording.CategoryName(category.Value).ToLowerInvariant()}." +
                  (category == ImageCategory.Chemical ? " A chemist must still check it: this is a classification, not a verified structure." : "");
        }
        catch (Exception ex) when (ex is InvalidOperationException or WorkerException or WorkerUnavailableException)
        {
            Notice = "Could not change the category: " + ex.Message;
        }
        finally
        {
            IsBusy = false;
            SyncCategoryChoice();
            RefreshSelection();
        }
    }

    private void SyncCategoryChoice()
    {
        _syncingCategory = true;
        try
        {
            var category = SelectedItem?.Job.Category;
            SelectedCategoryChoice = CategoryChoices.FirstOrDefault(c => c.Category == category);
        }
        finally
        {
            _syncingCategory = false;
        }
    }

    private bool Matches(ImageItemViewModel item) => InSelectedGroup(item) && Filter switch
    {
        GalleryFilter.ToDecide => IsUndecided(item),
        GalleryFilter.NeedsReview => item.Job.NeedsReview && item.Job.Status == ProcessingStatus.Completed,
        GalleryFilter.Approved => item.IsApproved,
        GalleryFilter.KeptOrSkipped => item.Job.Status == ProcessingStatus.Skipped || item.Job.Approval == ApprovalStatus.KeepOriginal,
        GalleryFilter.Failed => item.Job.Status == ProcessingStatus.Failed,
        _ => true,
    };

    private void RebuildVisible()
    {
        VisibleItems.Clear();
        foreach (var item in Items.Where(Matches))
        {
            VisibleItems.Add(item);
        }
    }

    private void UpdateSummary()
    {
        EligibleAiCount = Items.Where(i => JobOrchestrator.IsAiEligible(i.Job)).Select(i => i.Job.ResourceRef).Distinct().Count();
        int approved = 0, toDecide = 0, review = 0;
        foreach (var item in Items)
        {
            if (item.IsApproved)
            {
                approved++;
            }
            else if (item.CanApprove && item.Job.Approval == ApprovalStatus.Undecided)
            {
                toDecide++;
                if (item.Job.NeedsReview)
                {
                    review++;
                }
            }
        }

        DecisionSummary = $"{approved:N0} approved · {toDecide:N0} to decide" + (review > 0 ? $" ({review:N0} need a careful look)" : "");
        ApprovedCount = approved;
    }

    [ObservableProperty] public partial int ApprovedCount { get; set; }

    [RelayCommand]
    private Task Approve(ImageItemViewModel? item) => Decide(item, ApprovalStatus.Approved);

    [RelayCommand]
    private Task KeepOriginal(ImageItemViewModel? item) => Decide(item, ApprovalStatus.KeepOriginal);

    /// <summary>
    /// Records a decision. Deciding the image on screen moves on to the next image still to decide, so a long
    /// document can be reviewed with the A and K keys alone. Finished images can be decided while others are still processing.
    /// </summary>
    private async Task Decide(ImageItemViewModel? item, ApprovalStatus decision)
    {
        if (item is null || _cancelling)
        {
            return;
        }

        await _orchestrator.SetApprovalAsync([item.Id], decision);
        if (item == SelectedItem)
        {
            // Decisions are read from the job itself: the rows may not show a decision made a moment ago yet.
            var current = _orchestrator.SnapshotImages().ToDictionary(j => j.Id, j => j.Approval);
            bool ToDecide(ImageItemViewModel i) =>
                i != item && i.CanApprove && current.GetValueOrDefault(i.Id) == ApprovalStatus.Undecided;
            var at = VisibleItems.IndexOf(item);
            SelectedItem = VisibleItems.Skip(at + 1).FirstOrDefault(ToDecide)
                           ?? VisibleItems.Take(Math.Max(at, 0)).FirstOrDefault(ToDecide)
                           ?? item;
        }
    }

    private static bool IsUndecided(ImageItemViewModel item) => item.CanApprove && item.Job.Approval == ApprovalStatus.Undecided;

    [RelayCommand]
    private async Task ApproveAllEligible()
    {
        var count = await _orchestrator.ApproveAllEligibleAsync();
        Notice = count == 0
            ? "No further images are eligible. Images that need review must be approved one by one."
            : $"Approved {count:N0} image{(count == 1 ? "" : "s")}. Images that need review were left for you to check.";
    }

    [RelayCommand]
    private async Task RejectAll()
    {
        await _orchestrator.RejectAllAsync();
        Notice = "All enhancements rejected: every original image will be kept.";
    }

    [ObservableProperty] public partial string? Notice { get; set; }

    // ------------------------------------------------------------------ selection, previews, manual adjustments

    [ObservableProperty] public partial ImageItemViewModel? SelectedItem { get; set; }

    partial void OnSelectedItemChanged(ImageItemViewModel? value)
    {
        PendingVersion = null;
        ComparePair = value?.UsesAi == true ? "original-ai" : "original-opencv";
        StructureChecksExpanded = false;
        ShowMismatches = false;
        ViewZoom = 0;
        ViewCenter = new Avalonia.Point(0.5, 0.5);
        var settings = value?.Job.Settings ?? new EnhancementSettings();
        ManualMode = settings.Mode == "manual";
        DenoiseStrength = settings.DenoiseStrength;
        ContrastStrength = settings.ContrastStrength;
        SharpenAmount = settings.SharpenAmount;
        SelectedStrength = Wording.LevelName(settings.Strength);
        SelectedBold = Wording.BoldChoice(settings.BoldText);
        if (value is not null)
        {
            RememberLevel(value.Job);
        }

        RefreshSelection();
    }

    private void RefreshSelection()
    {
        if (SelectedCategoryChoice?.Category != SelectedItem?.Job.Category)
        {
            SyncCategoryChoice();
        }

        OnPropertyChanged(nameof(OriginalPreview));
        OnPropertyChanged(nameof(EnhancedPreview));
        OnPropertyChanged(nameof(AiPreview));
        OnPropertyChanged(nameof(LeftPreview));
        OnPropertyChanged(nameof(RightPreview));
        OnPropertyChanged(nameof(HasRightPreview));
        OnPropertyChanged(nameof(LeftLabel));
        OnPropertyChanged(nameof(RightLabel));
        OnPropertyChanged(nameof(RightMissingText));
        OnPropertyChanged(nameof(CanTryAiSelected));
        if (ComparePair.Contains("ai") && SelectedItem?.HasAiResult != true)
        {
            ComparePair = "original-opencv";
        }
        OnPropertyChanged(nameof(ReferenceSize));
        OnPropertyChanged(nameof(HasEnhancedPreview));
        OnPropertyChanged(nameof(HasSelection));
        OnPropertyChanged(nameof(CanAdjustSelection));
        OnPropertyChanged(nameof(LevelHistoryText));
        OnPropertyChanged(nameof(HasLevelHistory));
        OnPropertyChanged(nameof(LevelWarning));
        OnPropertyChanged(nameof(HasLevelWarning));
        OnPropertyChanged(nameof(SelectedMarks));
    }

    /// <summary>Show the numbered mismatch boxes on the original picture.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedMarks))]
    public partial bool ShowMismatches { get; set; }

    [ObservableProperty]
    public partial bool StructureChecksExpanded { get; set; }

    partial void OnStructureChecksExpandedChanged(bool value)
    {
        if (!value)
        {
            ShowMismatches = false;
        }
    }

    // Mismatch boxes belong to the original picture: shown only when it is on the left.
    public IReadOnlyList<ImageMark>? SelectedMarks => ShowMismatches && ComparePair.StartsWith("original") ? SelectedItem?.MismatchMarks : null;

    public bool HasSelection => SelectedItem is not null;
    public bool CanAdjustSelection => SelectedItem is { CanReenhance: true } && IsReady && !IsBusy && !IsAiRunning;

    /// <summary>Full-resolution original (decoded at most 1600 px wide); falls back to the thumbnail.</summary>
    public Bitmap? OriginalPreview => Preview(SelectedItem?.OriginalImagePath, SelectedItem?.Job.PixelSize?.Width, nameof(OriginalPreview))
                                      ?? SelectedItem?.OriginalThumbnail;

    public Bitmap? EnhancedPreview => Preview(SelectedItem?.EnhancedImagePath,
                                          (int?)((SelectedItem?.Job.PixelSize?.Width ?? 0) * (SelectedItem?.Job.Settings.ResizeFactor ?? 1)),
                                          nameof(EnhancedPreview))
                                      ?? SelectedItem?.EnhancedThumbnail;

    private Bitmap? Preview(string? path, int? width, string property) =>
        ImageCache.Previews.Get(path, () => Dispatcher.UIThread.Post(() =>
        {
            PreviewLoaded(property);
            OnPropertyChanged(nameof(ReferenceSize));
            OnPropertyChanged(nameof(HasEnhancedPreview));
        }), width);

    /// <summary>A picture finished loading: the viewers show it through the compared pair, so that pair updates too.</summary>
    private void PreviewLoaded(string property)
    {
        OnPropertyChanged(property);
        OnPropertyChanged(nameof(LeftPreview));
        OnPropertyChanged(nameof(RightPreview));
    }

    public bool HasEnhancedPreview => SelectedItem?.EnhancedImagePath is not null;

    public Bitmap? AiPreview => Preview(SelectedItem?.AiImagePath, SelectedItem?.Job.PixelSize?.Width, nameof(AiPreview))
                                ?? (SelectedItem?.AiThumbnailPath is { } thumb ? ImageCache.Thumbnails.Get(thumb, () => Dispatcher.UIThread.Post(() => PreviewLoaded(nameof(AiPreview)))) : null);

    // ------------------------------------------------------------------ compare: original, OpenCV result, AI version

    /// <summary>"original-opencv", "original-ai" or "opencv-ai": which two versions the viewers show.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LeftPreview), nameof(RightPreview), nameof(HasRightPreview), nameof(LeftLabel), nameof(RightLabel),
        nameof(RightMissingText), nameof(SelectedMarks), nameof(IsOriginalOpenCv), nameof(IsOriginalAi), nameof(IsOpenCvAi), nameof(IsRightAi))]
    public partial string ComparePair { get; set; } = "original-opencv";

    [RelayCommand]
    private void SetComparePair(string pair) => ComparePair = pair;

    public bool IsOriginalOpenCv => ComparePair == "original-opencv";
    public bool IsOriginalAi => ComparePair == "original-ai";
    public bool IsOpenCvAi => ComparePair == "opencv-ai";

    public Bitmap? LeftPreview => ComparePair == "opencv-ai" ? EnhancedPreview : OriginalPreview;
    public Bitmap? RightPreview => ComparePair.EndsWith("ai") ? AiPreview : HasEnhancedPreview ? EnhancedPreview : null;
    public bool HasRightPreview => ComparePair.EndsWith("ai") ? SelectedItem?.HasAiResult == true : HasEnhancedPreview;
    public string LeftLabel => ComparePair == "opencv-ai" ? NormalLabel : "Original";

    /// <summary>The AI version's label says whether it is in use, so it is clear which version will be saved.</summary>
    public string RightLabel => ComparePair.EndsWith("ai")
        ? SelectedItem?.Job.Ai is { } ai
            ? $"AI version ({ai.ModelNames})" + (!ai.Accepted ? " · not usable" : SelectedItem.Job.UseAiVersion ? " · in use" : " · not in use")
            : "AI version"
        : NormalLabel;

    /// <summary>The right picture is the AI version (its label is shown in the AI colour).</summary>
    public bool IsRightAi => ComparePair.EndsWith("ai");

    private string NormalLabel => "OpenCV result" + (SelectedItem?.Job is { Ai.Accepted: true, UseAiVersion: false } ? " · in use" : "");

    public string RightMissingText => ComparePair.EndsWith("ai") ? "No AI version yet" : "No OpenCV change: the original is kept";

    // ------------------------------------------------------------------ optional AI enhancement

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanAdjustSelection), nameof(CanTryAiSelected), nameof(CanTryAiOnEligible))]
    public partial bool IsAiRunning { get; set; }

    [ObservableProperty] public partial string? AiProgressText { get; set; }
    [ObservableProperty] public partial double AiProgressValue { get; set; }

    /// <summary>Pictures (each shared picture once) on which AI can be tried now.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanTryAiOnEligible), nameof(TryAiOnEligibleLabel))]
    public partial int EligibleAiCount { get; set; }

    public string TryAiOnEligibleLabel => $"Try AI on eligible images ({EligibleAiCount})";
    public bool CanTryAiOnEligible => IsReady && !IsBusy && !IsAiRunning && EligibleAiCount > 0;
    public bool CanTryAiSelected => IsReady && !IsBusy && !IsAiRunning && SelectedItem?.CanTryAi == true;

    private CancellationTokenSource? _aiStop;

    /// <summary>Tries AI on the selected picture; its AI version is then shown next to the OpenCV result.</summary>
    [RelayCommand]
    private async Task TryAi()
    {
        if (SelectedItem is not { } item || !CanTryAiSelected)
        {
            return;
        }

        await RunAiAsync([item.Id], single: true);
        if (SelectedItem == item && item.HasAiResult)
        {
            ComparePair = item.Job.Status == ProcessingStatus.Completed ? "opencv-ai" : "original-ai";
        }

        if (SelectedItem == item && item.Job is { Ai.Accepted: true, UseAiVersion: false, Approval: ApprovalStatus.Approved })
        {
            PendingVersion = "ai"; // approved earlier: ask now whether to use the AI version instead
        }
    }

    /// <summary>Tries AI on every eligible picture of the job, with progress; the others are left unchanged.</summary>
    [RelayCommand]
    private Task TryAiOnEligible() => CanTryAiOnEligible
        ? RunAiAsync(Items.Where(i => i.CanTryAi).Select(i => i.Id).ToList(), single: false)
        : Task.CompletedTask;

    [RelayCommand]
    private void StopAi() => _aiStop?.Cancel();

    /// <summary>"✓ AI enhancement applied to photo.png (SCUNet). ..." or why it was not applied.</summary>
    private string? SingleAiNotice(string id)
    {
        if (!_items.TryGetValue(id, out var item) || item.Job.Ai is not { } ai)
        {
            return null;
        }

        return item.Job.UseAiVersion
            ? $"✓ AI enhancement applied to {item.Title} ({ai.ModelNames}). Compare it with the {item.NormalVersionLabel}, then approve it or switch back."
            : $"{item.Title}: {Wording.AiSummary(item.Job)}";
    }

    private static string WholeJobAiNotice(AiRunSummary summary)
    {
        static string Pictures(int n) => $"{n:N0} picture{(n == 1 ? "" : "s")}";
        var parts = new List<string> { $"AI enhancement applied to {Pictures(summary.Applied)}" };
        if (summary.Accepted > summary.Applied)
        {
            parts.Add($"{summary.Accepted - summary.Applied:N0} ready but not applied because you had already decided {(summary.Accepted - summary.Applied == 1 ? "it" : "them")}");
        }

        if (summary.Rejected > 0)
        {
            parts.Add($"{summary.Rejected:N0} not applied because {(summary.Rejected == 1 ? "it" : "they")} did not pass the checks");
        }

        var failed = summary.Tried - summary.Accepted - summary.Rejected;
        if (failed > 0)
        {
            parts.Add($"{failed:N0} could not run");
        }

        var list = parts.Count == 1 ? parts[0] : string.Join(", ", parts.Take(parts.Count - 1)) + " and " + parts[^1];
        return $"AI tried on {Pictures(summary.Tried)}: {list}. Nothing was approved: check each AI version, then approve it or switch back." +
               (summary.Ineligible > 0 ? $" {Pictures(summary.Ineligible)} had no suitable AI repair and stayed as they were." : "");
    }

    private async Task RunAiAsync(IReadOnlyCollection<string> ids, bool single)
    {
        IsAiRunning = true;
        Notice = null;
        AiProgressValue = 0;
        AiProgressText = single ? "Trying AI on this picture…" : "Trying AI…";
        _aiStop = new CancellationTokenSource();
        var progress = new Progress<(int Done, int Total)>(p =>
        {
            AiProgressValue = p.Total == 0 ? 100 : 100.0 * p.Done / p.Total;
            if (!single)
            {
                AiProgressText = $"Trying AI: {p.Done:N0} of {p.Total:N0} pictures";
            }
        });
        try
        {
            var summary = await _orchestrator.TryAiAsync(ids, progress, _aiStop.Token);
            ApplyLatest(ids);
            Notice = single ? SingleAiNotice(ids.First()) : WholeJobAiNotice(summary);
        }
        catch (OperationCanceledException)
        {
            ApplyLatest(ids);
            Notice = "AI stopped. Pictures already tried keep their AI version; the others are unchanged.";
        }
        catch (Exception ex) when (ex is InvalidOperationException or WorkerException or WorkerUnavailableException)
        {
            Notice = "AI could not be tried: " + ex.Message;
        }
        finally
        {
            _aiStop.Dispose();
            _aiStop = null;
            IsAiRunning = false;
            AiProgressText = null;
            RefreshSelection();
        }
    }

    /// <summary>The version waiting for the user's confirmation ("ai" or "opencv"), because the picture was approved.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPendingVersion), nameof(PendingVersionText))]
    public partial string? PendingVersion { get; set; }

    public bool HasPendingVersion => PendingVersion is not null;

    public string PendingVersionText => PendingVersion is null || SelectedItem is not { } item ? ""
        : $"You approved the {(item.UsesAi ? "AI version" : item.NormalVersionLabel)} of this picture. Use the " +
          $"{(PendingVersion == "ai" ? "AI version" : item.NormalVersionLabel)} instead? It will then need your approval again.";

    /// <summary>Chooses the version to keep: "ai" or "opencv". An approved picture asks first.</summary>
    [RelayCommand]
    private async Task UseVersion(string which)
    {
        if (SelectedItem is not { } item || IsAiRunning)
        {
            return;
        }

        try
        {
            if (await _orchestrator.ChooseVersionAsync(item.Id, which == "ai"))
            {
                ApplyLatest([item.Id]);
                PendingVersion = null;
                ComparePair = which == "ai" ? "original-ai" : "original-opencv";
                Notice = which == "ai" ? "The AI version is used for this picture. Compare it closely before approving." : $"The {item.NormalVersionLabel} is used for this picture.";
            }
            else
            {
                PendingVersion = which; // approved: ask before changing what was approved
            }
        }
        catch (InvalidOperationException ex)
        {
            Notice = ex.Message;
        }
        finally
        {
            RefreshSelection();
        }
    }

    [RelayCommand]
    private async Task ConfirmVersionSwitch()
    {
        if (SelectedItem is not { } item || PendingVersion is not { } which)
        {
            return;
        }

        await _orchestrator.ChooseVersionAsync(item.Id, which == "ai", approvalChangeConfirmed: true);
        ApplyLatest([item.Id]);
        PendingVersion = null;
        ComparePair = which == "ai" ? "original-ai" : "original-opencv";
        Notice = "Version changed. The picture needs your approval again.";
        RefreshSelection();
    }

    [RelayCommand]
    private void CancelVersionSwitch() => PendingVersion = null;

    /// <summary>
    /// Shows the job's current state of these pictures (and every place they appear) at once, without waiting for the
    /// queued updates, so what follows a command (which versions to compare, the version in use) reads the new state.
    /// </summary>
    private void ApplyLatest(IReadOnlyCollection<string> ids)
    {
        var images = _orchestrator.SnapshotImages();
        var refs = images.Where(j => ids.Contains(j.Id)).Select(j => j.ResourceRef).ToHashSet();
        foreach (var image in images.Where(j => refs.Contains(j.ResourceRef)))
        {
            ApplyImage(image);
        }
    }

    /// <summary>Both images are drawn over the enhanced image's pixel area, so originals line up with results.</summary>
    public Avalonia.Size ReferenceSize
    {
        get
        {
            var size = SelectedItem?.Job.OutputPixelSize ?? SelectedItem?.Job.PixelSize;
            return size is null ? default : new Avalonia.Size(size.Width, size.Height);
        }
    }

    // ------------------------------------------------------------------ zoom viewer

    /// <summary>side (side by side) or slider (before/after wipe).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSideBySide), nameof(IsSlider))]
    public partial string ViewMode { get; set; } = "side";

    public bool IsSideBySide => ViewMode != "slider";
    public bool IsSlider => ViewMode == "slider";

    [RelayCommand]
    private void SetViewMode(string mode) => ViewMode = mode;

    /// <summary>Screen pixels per image pixel; 0 = fit to view. Shared by all viewers so they stay in sync.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ZoomText))]
    public partial double ViewZoom { get; set; }

    [ObservableProperty] public partial Avalonia.Point ViewCenter { get; set; } = new(0.5, 0.5);

    [ObservableProperty] public partial double SplitPosition { get; set; } = 0.5;

    public string ZoomText => ViewZoom <= 0 ? "Fit" : $"{ViewZoom * 100:0} %";

    [RelayCommand]
    private void ZoomFit()
    {
        ViewZoom = 0;
        ViewCenter = new Avalonia.Point(0.5, 0.5);
    }

    /// <summary>Gallery hidden, so the comparison gets the full width.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsGalleryVisible), nameof(DetailColumn), nameof(DetailColumnSpan), nameof(ExpandLabel))]
    public partial bool IsExpanded { get; set; }

    public bool IsGalleryVisible => !IsExpanded;
    public int DetailColumn => IsExpanded ? 0 : 2;
    public int DetailColumnSpan => IsExpanded ? 3 : 1;
    public string ExpandLabel => IsExpanded ? "Show list" : "Full width";

    [RelayCommand]
    private void ToggleExpanded() => IsExpanded = !IsExpanded;

    [RelayCommand]
    private void SelectNext(string direction)
    {
        if (VisibleItems.Count == 0)
        {
            return;
        }

        var index = SelectedItem is null ? -1 : VisibleItems.IndexOf(SelectedItem);
        index = Math.Clamp(index + (direction == "previous" ? -1 : 1), 0, VisibleItems.Count - 1);
        SelectedItem = VisibleItems[index];
    }

    [ObservableProperty] public partial bool ManualMode { get; set; }
    [ObservableProperty] public partial double DenoiseStrength { get; set; }
    [ObservableProperty] public partial double ContrastStrength { get; set; }
    [ObservableProperty] public partial double SharpenAmount { get; set; }

    public IReadOnlyList<string> StrengthChoices { get; } = Wording.LevelKeys.Select(Wording.LevelName).ToList();

    [ObservableProperty] public partial string? SelectedStrength { get; set; }

    public IReadOnlyList<string> BoldChoices => Wording.BoldChoices;

    [ObservableProperty] public partial string? SelectedBold { get; set; }

    /// <summary>Accuracy of every level tried on each image this session, so levels can be compared.</summary>
    private readonly Dictionary<string, Dictionary<string, (AccuracyRating Rating, double EdgesKept)>> _levelResults = [];

    private void RememberLevel(ImageJob job)
    {
        if (job.Status != ProcessingStatus.Completed || job.Fidelity is not { } f || job.Settings.Mode != "auto")
        {
            return;
        }

        if (!_levelResults.TryGetValue(job.ResourceRef, out var levels))
        {
            _levelResults[job.ResourceRef] = levels = [];
        }

        levels[job.Settings.Strength] = (ReportBuilder.Rate(job), f.EdgesKept);
    }

    /// <summary>
    /// The levels tried on the selected picture, one per line ("2 Standard: accuracy Good, 87.8 % of lines and
    /// letters kept"), shown from the history (clock) button next to the level buttons.
    /// </summary>
    public string LevelHistoryText
    {
        get
        {
            if (SelectedItem?.Job is not { } job || !_levelResults.TryGetValue(job.ResourceRef, out var levels) || levels.Count < 2)
            {
                return "";
            }

            return string.Join(Environment.NewLine, Wording.LevelKeys.Where(levels.ContainsKey).Select(k =>
                $"{Wording.LevelKeys.ToList().IndexOf(k) + 1} {Wording.LevelName(k)}: accuracy {levels[k].Rating}, " +
                $"{levels[k].EdgesKept * 100:0.0} % of lines and letters kept" + (k == job.Settings.Strength ? " (shown now)" : "")));
        }
    }

    public bool HasLevelHistory => LevelHistoryText.Length > 0;

    /// <summary>Shown on the page only when it matters: a level tried before kept more of the original.</summary>
    public string LevelWarning
    {
        get
        {
            if (SelectedItem?.Job is not { } job || !_levelResults.TryGetValue(job.ResourceRef, out var levels) || levels.Count < 2
                || !levels.TryGetValue(job.Settings.Strength, out var current))
            {
                return "";
            }

            return levels.Any(l => l.Value.Rating < current.Rating || l.Value.EdgesKept > current.EdgesKept + 0.02)
                ? "A lower level kept more of the original for this picture. Compare closely before approving (see the level history)."
                : "";
        }
    }

    public bool HasLevelWarning => LevelWarning.Length > 0;

    /// <summary>The level buttons: enhance the selected image again at level 1-4 (automatic mode).</summary>
    [RelayCommand]
    private async Task ApplyLevel(string key)
    {
        ManualMode = false;
        SelectedStrength = Wording.LevelName(key);
        await Reenhance();
    }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanAdjustSelection), nameof(CanReview))]
    public partial bool IsBusy { get; set; }

    [RelayCommand]
    private async Task Reenhance()
    {
        if (SelectedItem is not { } item || !CanAdjustSelection)
        {
            return;
        }

        var settings = new EnhancementSettings
        {
            Mode = ManualMode ? "manual" : "auto",
            Strength = Wording.LevelKey(SelectedStrength),
            BoldText = Wording.BoldValue(SelectedBold),
            TextCheck = item.Job.Settings.TextCheck,
            Redraw = item.Job.Settings.Redraw,
            RedrawAll = item.Job.Settings.RedrawAll,
            AiPhoto = item.Job.Settings.AiPhoto,
            AiRepair = item.Job.Settings.AiRepair,
            DenoiseStrength = Math.Round(DenoiseStrength, 1),
            ContrastStrength = Math.Round(ContrastStrength, 2),
            SharpenAmount = Math.Round(SharpenAmount, 2),
        };
        IsBusy = true;
        Notice = null;
        try
        {
            await _orchestrator.ReenhanceAsync(item.Id, settings);
            if (_orchestrator.SnapshotImages().FirstOrDefault(i => i.Id == item.Id) is { } updated)
            {
                RememberLevel(updated);
            }

            Notice = settings.Mode == "auto"
                ? $"Enhanced again at {Wording.Level(settings.Strength)}. Compare it zoomed in before approving."
                : "Enhanced again with the new settings. Review the result before approving.";
        }
        catch (Exception ex) when (ex is InvalidOperationException or WorkerException or WorkerUnavailableException)
        {
            Notice = "Could not enhance the image again: " + ex.Message;
        }
        finally
        {
            IsBusy = false;
            RefreshSelection();
        }
    }

    [RelayCommand]
    private void ResetAdjustments()
    {
        ManualMode = false;
        DenoiseStrength = ContrastStrength = SharpenAmount = 0;
        SelectedStrength = StrengthChoices[1];
        SelectedBold = BoldChoices[0];
    }

    // ------------------------------------------------------------------ export

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanExport), nameof(ShowExportPanel))]
    public partial bool IsExporting { get; set; }

    [ObservableProperty] public partial double ExportProgress { get; set; }
    [ObservableProperty] public partial string? ExportOperation { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowExportPanel))]
    public partial string? ExportMessage { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowExportPanel))]
    public partial string? ExportError { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowExportPanel))]
    public partial bool NeedsExportPassword { get; set; }

    [ObservableProperty] public partial string? ExportPassword { get; set; }

    [ObservableProperty] public partial bool IsSigned { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowExportPanel))]
    public partial bool ShowSignedWarning { get; set; }

    private bool _signedConfirmed;

    public bool CanExport => IsReady && !IsBusy && !IsExporting && !IsAiRunning && ApprovedCount > 0;

    public bool ShowExportPanel => IsExporting || ExportMessage is not null || ExportError is not null || ShowSignedWarning || NeedsExportPassword;

    partial void OnApprovedCountChanged(int value) => OnPropertyChanged(nameof(CanExport));
    partial void OnIsReadyChanged(bool value)
    {
        OnPropertyChanged(nameof(CanExport));
        OnPropertyChanged(nameof(CanTryAiOnEligible));
        OnPropertyChanged(nameof(CanTryAiSelected));
    }

    partial void OnIsBusyChanged(bool value)
    {
        OnPropertyChanged(nameof(CanExport));
        OnPropertyChanged(nameof(CanTryAiOnEligible));
        OnPropertyChanged(nameof(CanTryAiSelected));
    }

    [RelayCommand]
    private async Task Export()
    {
        if (!CanExport)
        {
            return;
        }

        if (IsSigned && !_signedConfirmed)
        {
            ShowSignedWarning = true; // the user must confirm first
            return;
        }

        var state = _orchestrator.State!;
        string? destination;
        if (state.SourceType == SourceType.Image)
        {
            destination = await _dialogs.PickFolderAsync();
        }
        else
        {
            var extension = System.IO.Path.GetExtension(state.SourceFileName);
            var suggested = System.IO.Path.GetFileNameWithoutExtension(state.SourceFileName) + "_enhanced" + extension;
            destination = await _dialogs.PickSaveFileAsync(suggested, extension);
        }

        if (destination is null)
        {
            return;
        }

        ExportMessage = null;
        ExportError = null;
        ExportProgress = 0;
        ExportOperation = "Starting";
        IsExporting = true;
        try
        {
            var password = NeedsExportPassword ? ExportPassword : null;
            var result = await _orchestrator.ExportAsync(destination, password);
            NeedsExportPassword = false;
            ExportPassword = null;
            ExportProgress = 100;
            var saved = state.SourceType == SourceType.Image
                ? $"Saved {result.Files.Count:N0} enhanced image(s) to {destination}."
                : $"Saved {destination} with {result.ImagesReplaced:N0} enhanced image(s).";
            var problems = result.NotReplaced.Count == 0
                ? ""
                : $" {result.NotReplaced.Count:N0} approved image(s) could not be used and the original was kept: " +
                  string.Join("; ", result.NotReplaced.Take(3)) + (result.NotReplaced.Count > 3 ? "…" : "");
            ExportMessage = $"{saved} {result.Validation}{problems}";
        }
        catch (JobFailure failure) when (failure.Code is "password_required" or "wrong_password")
        {
            NeedsExportPassword = true;
            ExportError = failure.Code == "wrong_password"
                ? "The password is not correct."
                : "Enter the PDF password to export. It is only kept for this session.";
        }
        catch (JobFailure failure)
        {
            ExportError = failure.Message + " Nothing was saved; the original is unchanged.";
        }
        finally
        {
            IsExporting = false;
            RefreshDiagnostics();
        }
    }

    [RelayCommand]
    private async Task ConfirmSignedExport()
    {
        _signedConfirmed = true;
        ShowSignedWarning = false;
        await Export();
    }

    [RelayCommand]
    private void DismissSignedWarning() => ShowSignedWarning = false;

    /// <summary>Done with this job: remove its temporary files and return to the start page.</summary>
    [RelayCommand]
    private async Task Finish()
    {
        await _orchestrator.FinishAsync();
        _finished(new JobOutcome(JobOutcomeKind.Finished), this);
    }

    // ------------------------------------------------------------------ report

    /// <summary>Saves an HTML report: what was re-altered or kept unchanged, the steps applied, quality before and
    /// after, and how accurate each result is.</summary>
    [RelayCommand]
    private async Task SaveReport()
    {
        if (_orchestrator.State is not { } state || WorkspaceRoot is not { } root)
        {
            return;
        }

        var suggested = System.IO.Path.GetFileNameWithoutExtension(state.SourceFileName) + "_enhancement_report.html";
        var path = await _dialogs.PickSaveFileAsync(suggested, ".html");
        if (path is null)
        {
            return;
        }

        try
        {
            var snapshot = new JobState
            {
                JobId = state.JobId,
                SourceType = state.SourceType,
                SourceFileName = state.SourceFileName,
                Images = _orchestrator.SnapshotImages().ToList(),
            };
            var html = await Task.Run(() => ReportBuilder.Build(snapshot, root));
            await System.IO.File.WriteAllTextAsync(path, html);
            Notice = $"Report saved to {path}.";
        }
        catch (Exception ex) when (ex is System.IO.IOException or UnauthorizedAccessException)
        {
            Notice = "The report could not be saved: " + ex.Message;
        }
    }

    // ------------------------------------------------------------------ diagnostics

    [ObservableProperty] public partial string DiagnosticsText { get; set; } = "";

    [RelayCommand]
    private void RefreshDiagnostics()
    {
        if (_orchestrator.State is null)
        {
            return;
        }

        var d = _orchestrator.GetDiagnostics();
        static string Seconds(double s) => s >= 60 ? $"{s / 60:0.0} min" : $"{s:0.0} s";
        var lines = new List<string>
        {
            $"Images: {d.TotalImages:N0} total · {d.Enhanced:N0} enhanced · {d.Skipped:N0} kept unchanged · {d.Failed:N0} failed",
            "Time: " + (d.StageSeconds.Count == 0 ? "–" : string.Join(" · ", d.StageSeconds.Select(kv => $"{kv.Key} {Seconds(kv.Value)}"))),
            $"Average enhancement time per image: {(d.AverageEnhanceSeconds is { } avg ? Seconds(avg) : "–")}",
            $"Peak worker memory: {d.PeakWorkerMemoryMegabytes:N0} MB · temporary disk use: {d.TempDiskMegabytes:N0} MB",
            $"Workers: {d.WorkerCount} · restarts: {d.WorkerRestarts} · batches: {d.Batches}",
        };
        if (d.SkipReasons.Count > 0)
        {
            lines.Add("Kept unchanged because: " + string.Join(", ", d.SkipReasons.Select(kv => $"{kv.Key} ({kv.Value})")));
        }

        if (d.FailureReasons.Count > 0)
        {
            lines.Add("Failures: " + string.Join(", ", d.FailureReasons.Select(kv => $"{kv.Key} ({kv.Value})")));
        }

        DiagnosticsText = string.Join(Environment.NewLine, lines);
    }

    private bool _disposed;

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        await StopForShutdownAsync();
        await _orchestrator.DisposeAsync();
        _appClosing.Dispose();
    }
}
