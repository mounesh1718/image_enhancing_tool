using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

public class WorkspaceManagerTests : IDisposable
{
    private readonly TempDirectory _root = new();
    private readonly WorkspaceManager _manager;

    public WorkspaceManagerTests()
    {
        _manager = new WorkspaceManager(_root.Path);
    }

    public void Dispose() => _root.Dispose();

    [Fact]
    public void Create_MakesAllSubfolders_WithUniqueIds()
    {
        var a = _manager.Create();
        var b = _manager.Create();

        Assert.NotEqual(a.JobId, b.JobId);
        foreach (var folder in Enum.GetValues<WorkspaceFolder>())
        {
            Assert.True(Directory.Exists(a.GetFolder(folder)), folder.ToString());
        }
    }

    [Theory]
    [InlineData("..")]
    [InlineData("../escape.png")]
    [InlineData("sub/file.png")]
    [InlineData("sub\\file.png")]
    [InlineData("C:\\Windows\\win.ini")]
    [InlineData("")]
    [InlineData("bad|name.png")]
    public void Resolve_RejectsNamesThatEscapeOrAreInvalid(string name)
    {
        var ws = _manager.Create();
        Assert.Throws<ArgumentException>(() => ws.Resolve(WorkspaceFolder.Enhanced, name));
    }

    [Fact]
    public void Resolve_ReturnsPathInsideFolder()
    {
        var ws = _manager.Create();
        var path = ws.Resolve(WorkspaceFolder.Enhanced, "img_0001.png");

        Assert.Equal(Path.Combine(ws.GetFolder(WorkspaceFolder.Enhanced), "img_0001.png"), path);
        Assert.Equal(Path.Combine("enhanced", "img_0001.png"), ws.ToRelative(path));
    }

    [Fact]
    public void ToRelative_RejectsOutsidePaths()
    {
        var ws = _manager.Create();
        Assert.Throws<ArgumentException>(() => ws.ToRelative(Path.Combine(_root.Path, "other.txt")));
    }

    [Fact]
    public void ListOpenDelete_RoundTrip()
    {
        var ws = _manager.Create();
        File.WriteAllText(ws.Resolve(WorkspaceFolder.Extracted, "a.bin"), "data");

        Assert.Contains(_manager.ListExisting(), w => w.JobId == ws.JobId);
        Assert.Equal(4, _manager.Open(ws.JobId)!.GetSizeBytes());

        _manager.Delete(ws);

        Assert.False(Directory.Exists(ws.RootPath));
        Assert.Null(_manager.Open(ws.JobId));
    }

    [Fact]
    public void ListExisting_IgnoresUnrelatedDirectories()
    {
        Directory.CreateDirectory(Path.Combine(_root.Path, "Job_not-a-guid"));
        Directory.CreateDirectory(Path.Combine(_root.Path, "UserStuff"));

        Assert.Empty(_manager.ListExisting());
    }

    [Fact]
    public void Open_RejectsTraversalIds()
    {
        Assert.Null(_manager.Open("..\\..\\Windows"));
    }

    [Fact]
    public void GetAvailableFreeSpace_IsPositive()
    {
        Assert.True(_manager.GetAvailableFreeSpace() > 0);
    }
}
