using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using ImageQualityEnhancer.ViewModels;

namespace ImageQualityEnhancer.Views;

public partial class MainWindow : Window, IFileDialogs
{
    private bool _shutdownDone;

    public MainWindow()
    {
        InitializeComponent();
        Closing += OnClosing;
        DragDrop.SetAllowDrop(this, true);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        var onStartPage = DataContext is MainViewModel { CurrentPage: StartViewModel };
        e.DragEffects = onStartPage && e.DataTransfer.TryGetFiles() is { Length: > 0 } ? DragDropEffects.Copy : DragDropEffects.None;
    }

    /// <summary>Dropping files on the start page starts a job, like choosing them with the button.</summary>
    private void OnDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is MainViewModel { CurrentPage: StartViewModel start } && e.DataTransfer.TryGetFiles() is { } items)
        {
            start.StartWithFiles(items.Select(f => f.TryGetLocalPath()).OfType<string>().ToList());
        }
    }

    /// <summary>Stops a running job before the window closes, so its progress is saved for later.</summary>
    private async void OnClosing(object? sender, WindowClosingEventArgs e)
    {
        if (_shutdownDone || DataContext is not MainViewModel vm)
        {
            return;
        }

        e.Cancel = true;
        _shutdownDone = true;
        await vm.ShutdownAsync();
        Close();
    }

    private static readonly FilePickerFileType Supported = new("PDF, Word documents and images")
    {
        Patterns = ["*.pdf", "*.docx", "*.docm", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif", "*.jp2", "*.ppm", "*.pgm"],
    };

    public async Task<IReadOnlyList<string>> PickInputFilesAsync()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Choose one PDF or Word document, or one or more images",
            AllowMultiple = true,
            FileTypeFilter = [Supported, FilePickerFileTypes.All],
        });
        return files.Select(f => f.TryGetLocalPath()).OfType<string>().ToList();
    }

    public async Task<string?> PickSaveFileAsync(string suggestedName, string extension)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save the enhanced document",
            SuggestedFileName = suggestedName,
            DefaultExtension = extension.TrimStart('.'),
            ShowOverwritePrompt = true,
        });
        return file?.TryGetLocalPath();
    }

    public async Task<string?> PickFolderAsync()
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose a folder for the enhanced images",
            AllowMultiple = false,
        });
        return folders.FirstOrDefault()?.TryGetLocalPath();
    }
}
