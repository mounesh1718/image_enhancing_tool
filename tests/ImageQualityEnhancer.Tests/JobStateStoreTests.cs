using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

public class JobStateStoreTests : IDisposable
{
    private readonly TempDirectory _root = new();
    private readonly JobWorkspace _workspace;

    public JobStateStoreTests()
    {
        _workspace = new WorkspaceManager(_root.Path).Create();
    }

    public void Dispose() => _root.Dispose();

    private static JobState SampleState() => new()
    {
        JobId = "job1",
        SourceType = SourceType.Pdf,
        SourceFileName = "Pharmaceutical_Report.pdf",
        Stage = JobStage.Enhancing,
        Images =
        [
            new ImageJob
            {
                Id = "img_0001",
                DocumentId = "job1",
                SourceType = SourceType.Pdf,
                ResourceRef = "hash-12",
                ResourceId = "r00001",
                Location = new ImageLocation { Xref = 12, Placements = [new PdfPlacement(3, [10, 20, 110, 220], false, false)] },
                PixelSize = new PixelSize(800, 600),
                DisplaySize = new DisplaySize(100, 200),
                Status = ProcessingStatus.Completed,
                Approval = ApprovalStatus.Approved,
                EnhancedPath = "enhanced/img_0001.png",
            },
            new ImageJob
            {
                Id = "img_0002",
                DocumentId = "job1",
                SourceType = SourceType.Pdf,
                ResourceRef = "hash-13",
                ResourceId = "r00002",
                Location = new ImageLocation { Xref = 13 },
                Status = ProcessingStatus.Processing,
            },
        ],
    };

    [Fact]
    public async Task SaveThenLoad_RoundTripsFields()
    {
        await JobStateStore.SaveAsync(_workspace, SampleState());
        var loaded = await JobStateStore.LoadAsync(_workspace);

        Assert.NotNull(loaded);
        Assert.Equal(JobStage.Enhancing, loaded.Stage);
        var first = loaded.Images[0];
        Assert.Equal(new PixelSize(800, 600), first.PixelSize);
        Assert.Equal(new double[] { 10, 20, 110, 220 }, first.Location.Placements![0].Rect);
        Assert.Equal(12, first.Location.Xref);
        Assert.Equal(ApprovalStatus.Approved, first.Approval);
        Assert.Equal(ProcessingStatus.Completed, first.Status);
    }

    [Fact]
    public async Task Load_ResetsInterruptedImagesToPending()
    {
        await JobStateStore.SaveAsync(_workspace, SampleState());
        var loaded = await JobStateStore.LoadAsync(_workspace);

        Assert.Equal(ProcessingStatus.Pending, loaded!.Images[1].Status);
    }

    [Fact]
    public async Task Save_UsesReadableStatusNames_AndLeavesNoTempFile()
    {
        var state = SampleState();
        await JobStateStore.SaveAsync(_workspace, state);
        state.Images[1].Approval = ApprovalStatus.KeepOriginal;
        await JobStateStore.SaveAsync(_workspace, state); // overwrite path

        var metadata = _workspace.GetFolder(WorkspaceFolder.Metadata);
        Assert.Equal(["queue.json"], Directory.GetFiles(metadata).Select(Path.GetFileName));
        var json = await File.ReadAllTextAsync(Path.Combine(metadata, "queue.json"));
        Assert.Contains("\"COMPLETED\"", json);
        Assert.Contains("\"KEEP_ORIGINAL\"", json);
    }

    [Fact]
    public async Task Load_ReturnsNullWhenNoQueue()
    {
        Assert.Null(await JobStateStore.LoadAsync(_workspace));
    }
}
