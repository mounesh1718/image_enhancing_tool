using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.ViewModels;

/// <summary>Switches between the start page and a running or finished job.</summary>
public partial class MainViewModel : ViewModelBase
{
    private readonly WorkspaceManager _workspaces;
    private readonly ProcessingOptions _options;

    public MainViewModel(WorkspaceManager workspaces, IFileDialogs dialogs, ProcessingOptions? options = null)
    {
        _workspaces = workspaces;
        _options = options ?? new ProcessingOptions();
        Dialogs = dialogs;
        Start = new StartViewModel(workspaces, dialogs, StartJob, ResumeJob);
        CurrentPage = Start;
        _ = Start.RefreshAsync();
    }

    public IFileDialogs Dialogs { get; }
    public StartViewModel Start { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsJobOpen))]
    public partial ViewModelBase CurrentPage { get; set; }

    public bool IsJobOpen => CurrentPage is JobViewModel;

    private void StartJob(IReadOnlyList<string> files, EnhancementSettings settings)
    {
        var name = files.Count == 1 ? Path.GetFileName(files[0]) : $"{files.Count} images";
        var options = _options with
        {
            DefaultSettings = _options.DefaultSettings with
            {
                Strength = settings.Strength, BoldText = settings.BoldText, TextCheck = settings.TextCheck, Redraw = settings.Redraw,
                AiPhoto = settings.AiPhoto, AiRepair = settings.AiRepair, RedrawAll = settings.RedrawAll,
            },
            // The structure reader needs about 0.8 GB per worker: redraw jobs use one worker.
            WorkerCount = settings.Redraw ? 1 : _options.WorkerCount,
        };
        var job = new JobViewModel(new JobOrchestrator(_workspaces, options), Dialogs, name, OnJobFinished);
        CurrentPage = job;
        job.Start(new JobRequest(files));
    }

    private void ResumeJob(JobWorkspace workspace, string? password)
    {
        var job = new JobViewModel(new JobOrchestrator(_workspaces, _options), Dialogs, "Resuming…", OnJobFinished);
        CurrentPage = job;
        job.Resume(workspace, password);
    }

    private async void OnJobFinished(JobOutcome outcome, JobViewModel job)
    {
        switch (outcome.Kind)
        {
            case JobOutcomeKind.Finished:
                await CloseAsync(job, $"Finished {job.DocumentName}. Its temporary files were removed.");
                break;

            case JobOutcomeKind.Cancelled:
                await CloseAsync(job, "The job was cancelled and its temporary files were removed. The original was not touched.");
                break;

            case JobOutcomeKind.Failed when outcome.ErrorCode is "password_required" or "wrong_password":
                var workspace = job.Workspace;
                await CloseAsync(job, null);
                if (workspace is not null)
                {
                    Start.AskForPassword(workspace, job.DocumentName, outcome.ErrorCode == "wrong_password");
                }

                break;

            case JobOutcomeKind.Failed:
                // Nothing usable was produced; remove the temporary copy and explain what happened.
                if (job.Workspace is { } failed && Directory.Exists(failed.RootPath))
                {
                    _workspaces.Delete(failed);
                }

                await CloseAsync(job, (outcome.Message ?? "The job stopped.") + " The original was not changed.", error: true);
                break;
        }
    }

    [RelayCommand]
    private async Task BackToStart()
    {
        if (CurrentPage is JobViewModel job)
        {
            await CloseAsync(job, $"{job.DocumentName} is saved for later. Open History (top right) to continue it.");
        }
    }

    private async Task CloseAsync(JobViewModel job, string? message, bool error = false)
    {
        CurrentPage = Start;
        await job.DisposeAsync();
        Start.ShowMessage(message, error);
        await Start.RefreshAsync();
    }

    /// <summary>Called when the window closes: stops processing and saves progress so the job can be resumed.</summary>
    public async Task ShutdownAsync()
    {
        if (CurrentPage is JobViewModel job)
        {
            await job.DisposeAsync();
        }
    }
}
