using System;
using System.Collections.Generic;
using System.Linq;

namespace ImageQualityEnhancer.Services;

/// <summary>
/// Estimates time remaining from recent per-image durations. Returns null until enough images have been timed,
/// and while durations vary too much to be meaningful.
/// </summary>
public sealed class EtaEstimator(int minSamples = 20, int window = 50)
{
    private readonly Queue<double> _recent = new();

    public void Add(TimeSpan duration)
    {
        _recent.Enqueue(duration.TotalMilliseconds);
        while (_recent.Count > window)
        {
            _recent.Dequeue();
        }
    }

    public TimeSpan? Estimate(int remaining, int workerCount = 1)
    {
        if (remaining <= 0)
        {
            return TimeSpan.Zero;
        }

        if (_recent.Count < minSamples)
        {
            return null;
        }

        var mean = _recent.Average();
        var std = Math.Sqrt(_recent.Average(v => (v - mean) * (v - mean)));
        // Very uneven image sizes make any single number misleading.
        if (mean <= 0 || std / mean > 1.5)
        {
            return null;
        }

        return TimeSpan.FromMilliseconds(mean * remaining / Math.Max(1, workerCount));
    }
}
