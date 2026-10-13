using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageQualityEnhancer.Services;

/// <summary>
/// Creates, finds and deletes per-job temporary workspaces under a single root directory.
/// Never touches anything outside that root, so user files cannot be deleted by cleanup.
/// </summary>
public sealed class WorkspaceManager
{
    private const string JobPrefix = "Job_";

    public WorkspaceManager(string? rootPath = null)
    {
        RootPath = Path.GetFullPath(rootPath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ImageQualityEnhancer",
            "Temp"));
    }

    public string RootPath { get; }

    public JobWorkspace Create()
    {
        var jobId = Guid.NewGuid().ToString("N");
        var workspace = new JobWorkspace(jobId, Path.Combine(RootPath, JobPrefix + jobId));
        foreach (var folder in Enum.GetValues<WorkspaceFolder>())
        {
            Directory.CreateDirectory(workspace.GetFolder(folder));
        }

        return workspace;
    }

    public JobWorkspace? Open(string jobId)
    {
        if (!IsValidJobId(jobId))
        {
            return null;
        }

        var path = Path.Combine(RootPath, JobPrefix + jobId);
        return Directory.Exists(path) ? new JobWorkspace(jobId, path) : null;
    }

    /// <summary>Workspaces left on disk, e.g. after a crash. Offered to the user for resume or cleanup.</summary>
    public IReadOnlyList<JobWorkspace> ListExisting()
    {
        if (!Directory.Exists(RootPath))
        {
            return [];
        }

        return Directory.EnumerateDirectories(RootPath, JobPrefix + "*")
            .Select(dir => Path.GetFileName(dir)[JobPrefix.Length..])
            .Where(IsValidJobId)
            .Select(id => new JobWorkspace(id, Path.Combine(RootPath, JobPrefix + id)))
            .ToList();
    }

    public void Delete(JobWorkspace workspace)
    {
        var expected = Path.GetFullPath(Path.Combine(RootPath, JobPrefix + workspace.JobId));
        if (!IsValidJobId(workspace.JobId)
            || !string.Equals(Path.GetFullPath(workspace.RootPath), expected, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Refusing to delete a directory that is not a managed job workspace.");
        }

        if (Directory.Exists(expected))
        {
            Directory.Delete(expected, recursive: true);
        }
    }

    /// <summary>Free space on the drive holding the workspace root.</summary>
    public long GetAvailableFreeSpace()
    {
        Directory.CreateDirectory(RootPath);
        return new DriveInfo(Path.GetPathRoot(RootPath)!).AvailableFreeSpace;
    }

    private static bool IsValidJobId(string jobId) =>
        jobId.Length == 32 && jobId.All(Uri.IsHexDigit);
}
