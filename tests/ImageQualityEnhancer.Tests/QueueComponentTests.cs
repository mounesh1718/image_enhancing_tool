using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

public class BatchPlannerTests
{
    private const long MB = 1024 * 1024;

    [Fact]
    public void NextBatch_StopsAtMemoryBudget()
    {
        long[] sizes = [300 * MB, 300 * MB, 300 * MB, 300 * MB];
        Assert.Equal(3, BatchPlanner.NextBatch(sizes, s => s, 1000 * MB, 20).Count);
    }

    [Fact]
    public void NextBatch_StopsAtMaxCount()
    {
        var sizes = Enumerable.Repeat(1 * MB, 100);
        Assert.Equal(20, BatchPlanner.NextBatch(sizes, s => s, 10_000 * MB, 20).Count);
    }

    [Fact]
    public void NextBatch_AlwaysTakesAtLeastOneImage()
    {
        long[] sizes = [5000 * MB, 1 * MB];
        Assert.Equal([5000 * MB], BatchPlanner.NextBatch(sizes, s => s, 1000 * MB, 20));
    }

    [Fact]
    public void NextBatch_EmptyQueue()
    {
        Assert.Empty(BatchPlanner.NextBatch(Array.Empty<long>(), s => s, 1000 * MB, 20));
    }

    [Fact]
    public void Budget_UsesShareOfAvailableRam_WithinLimits()
    {
        Assert.Equal(1000 * MB, BatchPlanner.Budget(2000 * MB, 0.5, 2048 * MB, 1));       // half of free RAM
        Assert.Equal(2048 * MB, BatchPlanner.Budget(64_000 * MB, 0.5, 2048 * MB, 1));     // capped by per-image limit x workers
        Assert.Equal(4096 * MB, BatchPlanner.Budget(64_000 * MB, 0.5, 2048 * MB, 2));
        Assert.Equal(2048 * MB, BatchPlanner.Budget(null, 0.5, 2048 * MB, 1));            // unknown RAM
        Assert.Equal(384 * MB, BatchPlanner.Budget(100 * MB, 0.5, 2048 * MB, 1));         // scarce RAM: small budget, big images skipped
    }

    [Fact]
    public void PerImageLimit_SplitsBudgetAcrossWorkers()
    {
        Assert.Equal(1000 * MB, BatchPlanner.PerImageLimit(2000 * MB, 2048 * MB, 2));
        Assert.Equal(2048 * MB, BatchPlanner.PerImageLimit(8000 * MB, 2048 * MB, 1));
    }
}

public class PauseGateTests
{
    [Fact]
    public async Task WaitBlocksWhilePaused_AndReleasesOnResume()
    {
        var gate = new PauseGate();
        await gate.WaitIfPausedAsync(CancellationToken.None); // not paused: immediate

        gate.Pause("low disk");
        var wait = gate.WaitIfPausedAsync(CancellationToken.None);
        await Task.Delay(50);
        Assert.False(wait.IsCompleted);
        Assert.Equal("low disk", gate.Reason);

        gate.Resume();
        await wait.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(gate.IsPaused);
        Assert.Null(gate.Reason);
    }

    [Fact]
    public async Task WaitIsCancellable()
    {
        var gate = new PauseGate();
        gate.Pause();
        using var cts = new CancellationTokenSource(50);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => gate.WaitIfPausedAsync(cts.Token));
    }
}

public class EtaEstimatorTests
{
    [Fact]
    public void NoEstimateUntilEnoughSamples()
    {
        var eta = new EtaEstimator(minSamples: 20);
        for (var i = 0; i < 19; i++)
        {
            eta.Add(TimeSpan.FromSeconds(1));
        }

        Assert.Null(eta.Estimate(100));
        eta.Add(TimeSpan.FromSeconds(1));
        Assert.Equal(TimeSpan.FromSeconds(100), eta.Estimate(100));
        Assert.Equal(TimeSpan.FromSeconds(50), eta.Estimate(100, workerCount: 2));
    }

    [Fact]
    public void NoEstimateWhenDurationsVaryWildly()
    {
        var eta = new EtaEstimator(minSamples: 20);
        for (var i = 0; i < 30; i++)
        {
            eta.Add(TimeSpan.FromMilliseconds(i % 10 == 0 ? 60_000 : 50));
        }

        Assert.Null(eta.Estimate(10));
    }
}

public class ApprovalRulesTests
{
    private static ImageJob Image(ProcessingStatus status, bool review = false) => new()
    {
        Id = Guid.NewGuid().ToString("N"),
        DocumentId = "d",
        SourceType = SourceType.Pdf,
        Location = new ImageLocation(),
        ResourceRef = "r",
        ResourceId = "r1",
        Status = status,
        NeedsReview = review,
        EnhancedPath = status == ProcessingStatus.Completed ? "enhanced/r1.png" : null,
    };

    [Fact]
    public void OnlyCompletedImagesCanBeApproved()
    {
        Assert.True(ApprovalRules.Set(Image(ProcessingStatus.Completed), ApprovalStatus.Approved));
        foreach (var status in new[] { ProcessingStatus.Skipped, ProcessingStatus.Failed, ProcessingStatus.Pending })
        {
            var image = Image(status);
            Assert.False(ApprovalRules.Set(image, ApprovalStatus.Approved));
            Assert.Equal(ApprovalStatus.Undecided, image.Approval);
            Assert.True(ApprovalRules.Set(image, ApprovalStatus.KeepOriginal));
        }
    }

    [Fact]
    public void ApproveAllEligible_SkipsFlaggedAndKeepsEarlierDecisions()
    {
        var plain = Image(ProcessingStatus.Completed);
        var flagged = Image(ProcessingStatus.Completed, review: true);
        var rejected = Image(ProcessingStatus.Completed);
        rejected.Approval = ApprovalStatus.KeepOriginal;
        var failed = Image(ProcessingStatus.Failed);
        var all = new[] { plain, flagged, rejected, failed };

        Assert.Equal(1, ApprovalRules.ApproveAllEligible(all));
        Assert.Equal(ApprovalStatus.Approved, plain.Approval);
        Assert.Equal(ApprovalStatus.Undecided, flagged.Approval);
        Assert.Equal(ApprovalStatus.KeepOriginal, rejected.Approval);
        Assert.Equal([plain], ApprovalRules.ToReplace(all));
    }

    [Fact]
    public void RejectAll_KeepsEveryOriginal()
    {
        var images = new[] { Image(ProcessingStatus.Completed), Image(ProcessingStatus.Completed, review: true) };
        ApprovalRules.ApproveAllEligible(images);
        ApprovalRules.RejectAll(images);
        Assert.All(images, i => Assert.Equal(ApprovalStatus.KeepOriginal, i.Approval));
        Assert.Empty(ApprovalRules.ToReplace(images));
    }
}
