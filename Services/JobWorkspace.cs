using System;
using System.IO;

namespace ImageQualityEnhancer.Services;

public enum WorkspaceFolder
{
    Original,
    Extracted,
    Enhanced,
    Thumbnails,
    Metadata,
    Export,
}

/// <summary>A single job's temporary directory. All file access for the job goes through <see cref="Resolve"/>.</summary>
public sealed class JobWorkspace
{
    internal JobWorkspace(string jobId, string rootPath)
    {
        JobId = jobId;
        RootPath = rootPath;
    }

    public string JobId { get; }
    public string RootPath { get; }

    public string GetFolder(WorkspaceFolder folder) =>
        Path.Combine(RootPath, folder.ToString().ToLowerInvariant());

    /// <summary>
    /// Returns an absolute path for <paramref name="fileName"/> inside <paramref name="folder"/>.
    /// Throws if the name tries to escape the folder (rooted paths, "..", separators).
    /// </summary>
    public string Resolve(WorkspaceFolder folder, string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName)
            || fileName != Path.GetFileName(fileName)
            || fileName is "." or ".."
            || fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException($"Invalid workspace file name: '{fileName}'.", nameof(fileName));
        }

        var dir = Path.GetFullPath(GetFolder(folder));
        var full = Path.GetFullPath(Path.Combine(dir, fileName));
        if (!full.StartsWith(dir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Path escapes workspace: '{fileName}'.", nameof(fileName));
        }

        return full;
    }

    /// <summary>Converts an absolute path inside the workspace to a workspace-relative one for persistence.</summary>
    public string ToRelative(string absolutePath)
    {
        var root = Path.GetFullPath(RootPath) + Path.DirectorySeparatorChar;
        var full = Path.GetFullPath(absolutePath);
        if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Path is outside the workspace: '{absolutePath}'.", nameof(absolutePath));
        }

        return full[root.Length..];
    }

    public long GetSizeBytes()
    {
        if (!Directory.Exists(RootPath))
        {
            return 0;
        }

        long total = 0;
        foreach (var file in new DirectoryInfo(RootPath).EnumerateFiles("*", SearchOption.AllDirectories))
        {
            total += file.Length;
        }

        return total;
    }
}
