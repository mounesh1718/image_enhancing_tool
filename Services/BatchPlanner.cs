using System;
using System.Collections.Generic;

namespace ImageQualityEnhancer.Services;

/// <summary>
/// Splits pending images into batches sized by estimated memory rather than a fixed count.
///
/// A batch takes images in queue order while their combined estimate fits the memory budget and the batch is not
/// larger than <c>maxBatchSize</c>; it always takes at least one image. Because the whole batch fits the budget,
/// any subset of it can run in parallel on several workers without exceeding available RAM.
/// </summary>
public static class BatchPlanner
{
    public static List<T> NextBatch<T>(IEnumerable<T> pending, Func<T, long> estimate, long budgetBytes, int maxBatchSize)
    {
        var batch = new List<T>();
        long used = 0;
        foreach (var item in pending)
        {
            var need = Math.Max(0, estimate(item));
            if (batch.Count > 0 && (batch.Count >= maxBatchSize || used + need > budgetBytes))
            {
                break;
            }

            batch.Add(item);
            used += need;
        }

        return batch;
    }

    /// <summary>Smallest budget used when RAM is scarce: enough for ordinary document images, so only large ones are skipped.</summary>
    public const long MinimumBudgetBytes = 384L * 1024 * 1024;

    /// <summary>
    /// Memory a batch may use: a share of currently available RAM, never more than the per-image limit times the
    /// number of workers (one image per worker runs at a time). When RAM is scarce the budget shrinks, so large
    /// images are skipped with a reason instead of risking an out-of-memory failure.
    /// </summary>
    public static long Budget(long? availableRamBytes, double availableFraction, long maxImageMemoryBytes, int workerCount)
    {
        var ceiling = maxImageMemoryBytes * Math.Max(1, workerCount);
        if (availableRamBytes is not { } available || available <= 0)
        {
            return ceiling;
        }

        var share = (long)(available * Math.Clamp(availableFraction, 0.05, 0.9));
        return Math.Clamp(share, Math.Min(MinimumBudgetBytes, ceiling), ceiling);
    }

    /// <summary>Per-image memory limit sent to the worker: the smaller of the configured limit and the batch budget.</summary>
    public static long PerImageLimit(long budgetBytes, long maxImageMemoryBytes, int workerCount) =>
        Math.Min(maxImageMemoryBytes, Math.Max(budgetBytes / Math.Max(1, workerCount), 64L * 1024 * 1024));
}
