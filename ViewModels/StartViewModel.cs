using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.ViewModels;

/// <summary>A job left on disk (unfinished, or finished but not exported yet).</summary>
public sealed class SavedJobViewModel(JobWorkspace workspace, JobState state)
{
    public JobWorkspace Workspace { get; } = workspace;
    public JobState State { get; } = state;
    public string Name => State.SourceFileName;

    public string Description
    {
        get
        {
            var images = State.Images.Count;
            var done = State.Images.Count(i => i.Status is not (ProcessingStatus.Pending or ProcessingStatus.Processing));
            var stage = State.Stage switch
            {
                JobStage.AwaitingReview => "ready for review",
                JobStage.Failed => "stopped: " + State.Error,
                _ when State.ErrorCode is "password_required" or "wrong_password" => "waiting for the PDF password",
                _ => images == 0 ? "not started" : $"{done:N0} of {images:N0} images processed",
            };
            return $"{State.CreatedAt.LocalDateTime:g} · {stage}";
        }
    }

    public bool CanResume => State.Stage is not (JobStage.Failed or JobStage.Cancelled);
}

public sealed partial class StartViewModel : ViewModelBase
{
    private readonly WorkspaceManager _workspaces;
    private readonly IFileDialogs _dialogs;
    private readonly Action<IReadOnlyList<string>, EnhancementSettings> _start;
    private readonly Action<JobWorkspace, string?> _resume;

    public StartViewModel(WorkspaceManager workspaces, IFileDialogs dialogs,
        Action<IReadOnlyList<string>, EnhancementSettings> start, Action<JobWorkspace, string?> resume)
    {
        _workspaces = workspaces;
        _dialogs = dialogs;
        _start = start;
        _resume = resume;
    }

    public ObservableCollection<SavedJobViewModel> SavedJobs { get; } = [];

    /// <summary>Level 1-4 for every image of the next job (each image can be changed afterwards).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLevel1), nameof(IsLevel2), nameof(IsLevel3), nameof(IsLevel4), nameof(LevelDescription))]
    public partial string Level { get; set; } = "standard";

    public bool IsLevel1 => Level == "gentle";
    public bool IsLevel2 => Level == "standard";
    public bool IsLevel3 => Level == "strong";
    public bool IsLevel4 => Level == "maximum";

    public string LevelDescription => Level switch
    {
        "gentle" => "Small fixes only.",
        "strong" => "Stronger cleaning and deblurring of scans and faded pages. Faint text gets slightly bolder; photos get extra clarity (local contrast). Clean pictures keep their look.",
        "maximum" => "The strongest cleaning and the crispest text for scans and faded pages. Clean pictures (charts, tables, diagrams saved from software) keep their look. Check the results closely.",
        _ => "Recommended. Repairs only what is measured (noise, blur, JPEG blocks, fading, grey paper). Clean pictures, colours, lines and photos keep their look.",
    };

    [RelayCommand]
    private void SetLevel(string key) => Level = key;

    public IReadOnlyList<string> BoldChoices => Wording.BoldChoices;

    [ObservableProperty] public partial string SelectedBold { get; set; } = Wording.BoldChoices[0];

    /// <summary>OCR text check on images with text (about 4 s more per such image).</summary>
    [ObservableProperty] public partial bool CheckText { get; set; } = true;

    /// <summary>Improve quality only (re-alter), or also redraw chemical structures that can be proven.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsImproveOnly), nameof(IsRedrawProven), nameof(IsRedrawAll), nameof(StructuresDescription))]
    public partial bool RedrawStructures { get; set; }

    /// <summary>With <see cref="RedrawStructures"/>: redraw every structure the reader can read, unproven ones marked.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsRedrawProven), nameof(IsRedrawAll), nameof(StructuresDescription))]
    public partial bool RedrawAll { get; set; }

    public bool IsImproveOnly => !RedrawStructures;
    public bool IsRedrawProven => RedrawStructures && !RedrawAll;
    public bool IsRedrawAll => RedrawStructures && RedrawAll;

    public string StructuresDescription => IsRedrawAll
        ? "Every picture classified as a chemical diagram that the reader can read is redrawn. Redraws that do not pass every check are marked \"not verified\": compare every atom and bond before approving. Tables, graphs and photos are never redrawn."
        : RedrawStructures
            ? "A picture classified as a chemical diagram is redrawn only when every automatic check passes (marked \"verified\"). The checks do not prove it matches the original: you compare and approve every redraw."
            : "Recommended. Every image keeps its own drawing, only cleaner and sharper.";

    /// <summary>"improve", "redraw" (when proven) or "all".</summary>
    [RelayCommand]
    private void SetStructures(string choice)
    {
        RedrawStructures = choice is "redraw" or "all";
        RedrawAll = choice == "all";
    }

    /// <summary>AI clarity for photos (never used on text or structures).</summary>
    [ObservableProperty] public partial bool AiPhotos { get; set; }

    /// <summary>AI repair during processing (grain, JPEG blocks, blurred text). Off: AI is offered after processing, on
    /// the comparison screen, picture by picture or for all eligible pictures.</summary>
    [ObservableProperty] public partial bool AiRepair { get; set; }

    private EnhancementSettings ChosenSettings => new()
    {
        Strength = Level, BoldText = Wording.BoldValue(SelectedBold), TextCheck = CheckText, Redraw = RedrawStructures, RedrawAll = RedrawAll,
        AiPhoto = AiPhotos, AiRepair = AiRepair,
    };

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SavedJobCount))]
    public partial bool HasSavedJobs { get; set; }

    /// <summary>Jobs saved for later, for the badge on the History button.</summary>
    public int SavedJobCount => SavedJobs.Count;

    /// <summary>The History panel (jobs saved for later), opened from the clock button at the top right.</summary>
    [ObservableProperty] public partial bool IsHistoryOpen { get; set; }

    [RelayCommand]
    private void ToggleHistory() => IsHistoryOpen = !IsHistoryOpen;

    [RelayCommand]
    private void CloseHistory() => IsHistoryOpen = false;
    [ObservableProperty] public partial string? Message { get; set; }
    [ObservableProperty] public partial bool IsError { get; set; }

    // A job waiting for a PDF password.
    [ObservableProperty] public partial bool NeedsPassword { get; set; }
    [ObservableProperty] public partial string? PasswordPrompt { get; set; }
    [ObservableProperty] public partial string? Password { get; set; }
    private JobWorkspace? _waitingForPassword;

    [RelayCommand]
    private async Task ChooseFiles()
    {
        var files = await _dialogs.PickInputFilesAsync();
        if (files.Count > 0)
        {
            ShowMessage(null);
            NeedsPassword = false;
            _start(files, ChosenSettings);
        }
    }

    /// <summary>Starts a job from files dropped onto the window.</summary>
    public void StartWithFiles(IReadOnlyList<string> files)
    {
        if (files.Count == 0)
        {
            return;
        }

        ShowMessage(null);
        NeedsPassword = false;
        _start(files, ChosenSettings);
    }

    public void AskForPassword(JobWorkspace workspace, string documentName, bool wrong)
    {
        _waitingForPassword = workspace;
        Password = null;
        PasswordPrompt = wrong
            ? $"The password for {documentName} is not correct. Try again."
            : $"{documentName} is password-protected. Enter its password to continue.";
        NeedsPassword = true;
        // The password box stands for this job; it is not listed a second time below.
        foreach (var saved in SavedJobs.Where(j => j.Workspace.RootPath == workspace.RootPath).ToList())
        {
            SavedJobs.Remove(saved);
        }

        HasSavedJobs = SavedJobs.Count > 0;
        OnPropertyChanged(nameof(SavedJobCount));
    }

    [RelayCommand]
    private void SubmitPassword()
    {
        if (_waitingForPassword is { } workspace && !string.IsNullOrEmpty(Password))
        {
            NeedsPassword = false;
            var password = Password;
            Password = null;
            _resume(workspace, password);
        }
    }

    /// <summary>Closes the password box; the job stays in the list to continue or remove later.</summary>
    [RelayCommand]
    private async Task DismissPassword()
    {
        NeedsPassword = false;
        _waitingForPassword = null;
        Password = null;
        await RefreshAsync();
    }

    [RelayCommand]
    private void ResumeJob(SavedJobViewModel? job)
    {
        if (job is null || !job.CanResume)
        {
            return;
        }

        IsHistoryOpen = false;
        if (job.State.ErrorCode is "password_required" or "wrong_password")
        {
            AskForPassword(job.Workspace, job.Name, wrong: false);
            return;
        }

        _resume(job.Workspace, null);
    }

    [RelayCommand]
    private async Task DeleteJob(SavedJobViewModel? job)
    {
        if (job is null)
        {
            return;
        }

        _workspaces.Delete(job.Workspace);
        await RefreshAsync();
        ShowMessage($"Removed {job.Name} from History. Its temporary copies are deleted; the original document was not touched.");
    }

    public void ShowMessage(string? message, bool error = false)
    {
        Message = message;
        IsError = error;
    }

    public async Task RefreshAsync()
    {
        var found = new List<SavedJobViewModel>();
        foreach (var workspace in _workspaces.ListExisting())
        {
            try
            {
                if (await JobStateStore.LoadAsync(workspace) is { } state)
                {
                    found.Add(new SavedJobViewModel(workspace, state));
                }
            }
            catch (Exception ex) when (ex is System.IO.IOException or System.Text.Json.JsonException or UnauthorizedAccessException)
            {
                // A workspace without a readable queue is left for the user to delete manually.
            }
        }

        SavedJobs.Clear();
        foreach (var job in found.OrderByDescending(j => j.State.CreatedAt))
        {
            SavedJobs.Add(job);
        }

        HasSavedJobs = SavedJobs.Count > 0;
        OnPropertyChanged(nameof(SavedJobCount));
    }
}
