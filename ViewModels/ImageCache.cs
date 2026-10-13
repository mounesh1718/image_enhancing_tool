using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;

namespace ImageQualityEnhancer.ViewModels;

/// <summary>
/// A bounded, least-recently-used cache of decoded images, loaded off the UI thread.
///
/// Gallery rows never keep their own bitmap references; they ask the cache each time. With virtualisation only
/// visible rows ask, so memory stays bounded by the capacity however many images the document has. Evicted bitmaps
/// are not disposed (one may still be on screen); they are released by the garbage collector once unused.
/// Files are read fully into memory before decoding so the worker can overwrite them (re-enhance) without locks.
/// </summary>
public sealed class ImageCache(int capacity, int? decodeWidth = null)
{
    private readonly object _lock = new();
    private readonly Dictionary<string, LinkedListNode<(string Key, Bitmap? Bitmap)>> _entries = [];
    private readonly LinkedList<(string Key, Bitmap? Bitmap)> _order = new();
    private readonly Dictionary<string, Task<Bitmap?>> _loading = [];

    /// <summary>Thumbnails for the gallery (already small, decoded as-is).</summary>
    public static ImageCache Thumbnails { get; } = new(400);

    /// <summary>
    /// Previews of the selected image for the zoom viewer: full resolution up to 4096 px wide (zooming shows the real
    /// pixels), larger images decoded at that width to bound memory. Only the last few are kept.
    /// </summary>
    public static ImageCache Previews { get; } = new(4, decodeWidth: 4096);

    /// <summary>Returns the cached bitmap, or null and starts loading it; <paramref name="loaded"/> runs on completion.</summary>
    /// <param name="knownWidth">The image's pixel width if known; previews wider than the cache's limit are decoded smaller.</param>
    public Bitmap? Get(string? path, Action? loaded, int? knownWidth = null)
    {
        if (path is null)
        {
            return null;
        }

        var key = Key(path);
        if (key is null)
        {
            return null;
        }

        lock (_lock)
        {
            if (_entries.TryGetValue(key, out var node))
            {
                _order.Remove(node);
                _order.AddFirst(node);
                return node.Value.Bitmap;
            }

            if (_loading.ContainsKey(key))
            {
                return null;
            }

            var task = Task.Run(() => Decode(path, knownWidth));
            _loading[key] = task;
            task.ContinueWith(t =>
            {
                lock (_lock)
                {
                    _loading.Remove(key);
                    Add(key, t.IsCompletedSuccessfully ? t.Result : null);
                }

                loaded?.Invoke();
            }, TaskScheduler.Default);
            return null;
        }
    }

    private void Add(string key, Bitmap? bitmap)
    {
        if (_entries.Remove(key, out var existing))
        {
            _order.Remove(existing);
        }

        _entries[key] = _order.AddFirst((key, bitmap));
        while (_order.Count > capacity)
        {
            var last = _order.Last!;
            _order.RemoveLast();
            _entries.Remove(last.Value.Key);
        }
    }

    private Bitmap? Decode(string path, int? knownWidth)
    {
        try
        {
            using var stream = new MemoryStream(File.ReadAllBytes(path));
            // Shown at native size when it fits, so reviewers judge the real pixels; never upscaled.
            return decodeWidth is { } width && knownWidth > width
                ? Bitmap.DecodeToWidth(stream, width, BitmapInterpolationMode.HighQuality)
                : new Bitmap(stream);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException
                                       or InvalidOperationException or NotSupportedException)
        {
            return null; // formats the UI cannot decode (e.g. TIFF) fall back to the thumbnail
        }
    }

    /// <summary>Path plus modification time, so a re-enhanced file is not served from the cache.</summary>
    private static string? Key(string path)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists ? $"{path}|{info.LastWriteTimeUtc.Ticks}|{info.Length}" : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return null;
        }
    }
}
