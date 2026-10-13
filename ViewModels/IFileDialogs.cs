using System.Collections.Generic;
using System.Threading.Tasks;

namespace ImageQualityEnhancer.ViewModels;

/// <summary>File dialogs, implemented by the window so view models stay free of UI types.</summary>
public interface IFileDialogs
{
    /// <summary>Local paths of the chosen files, or an empty list if the user cancelled.</summary>
    Task<IReadOnlyList<string>> PickInputFilesAsync();

    /// <summary>Local path for the exported document, or null if the user cancelled.</summary>
    Task<string?> PickSaveFileAsync(string suggestedName, string extension);

    /// <summary>Local folder for exported images, or null if the user cancelled.</summary>
    Task<string?> PickFolderAsync();
}
