using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;

namespace ImageQualityEnhancer.Services.Docx;

public sealed record DocxValidationResult(bool Ok, IReadOnlyList<string> Problems, int EntriesCompared, int ImagesChecked);

/// <summary>Checks an exported .docx against its original before it is shown to the user as finished.</summary>
public static class DocxValidator
{
    private const int MaxProblems = 50;

    /// <param name="applied">(part, ordinal) -> enhanced file that must now be shown there.</param>
    /// <param name="changedEntries">Zip entries the exporter rewrote on purpose.</param>
    public static DocxValidationResult Validate(
        string originalPath,
        string outputPath,
        DocxScanResult originalScan,
        IReadOnlyDictionary<(string PartUri, int Ordinal), string> applied,
        IReadOnlySet<string> changedEntries,
        CancellationToken ct = default)
    {
        var problems = new List<string>();
        var compared = 0;
        var imagesChecked = 0;

        // 1. Package structure: every original entry is still there; untouched ones are byte-identical (CRC + size).
        using (var original = ZipFile.OpenRead(originalPath))
        using (var output = OpenOrReport(outputPath, problems))
        {
            if (output is null)
            {
                return new DocxValidationResult(false, problems, 0, 0);
            }

            foreach (var entry in original.Entries)
            {
                ct.ThrowIfCancellationRequested();
                var copy = output.GetEntry(entry.FullName);
                if (copy is null)
                {
                    Note($"{entry.FullName} is missing from the output");
                }
                else if (!changedEntries.Contains(entry.FullName) && (copy.Crc32 != entry.Crc32 || copy.Length != entry.Length))
                {
                    Note($"{entry.FullName} changed but was not meant to");
                }

                compared++;
            }
        }

        // 2. The output opens as a Word document and every image reference resolves as before.
        DocxScanResult outputScan;
        try
        {
            outputScan = DocxScanner.Scan(outputPath, ct: ct);
        }
        catch (DocxException ex)
        {
            problems.Add($"the output cannot be read: {ex.Message}");
            return new DocxValidationResult(false, problems, compared, 0);
        }

        foreach (var (part, count) in originalScan.PartElementCounts)
        {
            if (!outputScan.PartElementCounts.TryGetValue(part, out var outCount) || outCount != count)
            {
                Note($"{part}: document structure changed");
            }
            else if (outputScan.PartContentHashes[part] != originalScan.PartContentHashes[part])
            {
                Note($"{part}: text changed");
            }
        }

        var before = originalScan.Occurrences.ToDictionary(o => (o.PartUri, o.Ref.Ordinal));
        var after = outputScan.Occurrences.ToDictionary(o => (o.PartUri, o.Ref.Ordinal));
        if (before.Count != after.Count)
        {
            Note($"image count changed from {before.Count} to {after.Count}");
        }

        // 3. Each occurrence keeps its displayed size, crop and placement, and shows the right pixels:
        //    the enhanced file where approved, the untouched original everywhere else.
        using var originalHashes = new ImageHashes(originalPath);
        using var outputHashes = new ImageHashes(outputPath);
        var fileHashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, was) in before)
        {
            ct.ThrowIfCancellationRequested();
            if (problems.Count >= MaxProblems)
            {
                break;
            }

            if (!after.TryGetValue(key, out var now))
            {
                Note($"image {key.Ordinal} in {key.PartUri} is missing");
                continue;
            }

            var a = was.Ref;
            var b = now.Ref;
            if (a.Kind != b.Kind || a.ExtentCx != b.ExtentCx || a.ExtentCy != b.ExtentCy || a.Anchored != b.Anchored
                || a.InTable != b.InTable || a.DrawingId != b.DrawingId || a.HasSvg != b.HasSvg
                || !(a.Crop ?? []).SequenceEqual(b.Crop ?? []))
            {
                Note($"image {key.Ordinal} in {key.PartUri}: size, crop or placement changed");
            }

            if (was.ImageUri is null || now.ImageUri is null)
            {
                if (was.ImageUri != now.ImageUri)
                {
                    Note($"image {key.Ordinal} in {key.PartUri}: reference changed");
                }

                continue;
            }

            var expected = applied.TryGetValue(key, out var enhanced)
                ? fileHashes.TryGetValue(enhanced, out var h) ? h : fileHashes[enhanced] = HashFile(enhanced)
                : originalHashes.Get(was.ImageUri);
            if (outputHashes.Get(now.ImageUri) != expected)
            {
                Note(applied.ContainsKey(key)
                    ? $"image {key.Ordinal} in {key.PartUri} was not replaced"
                    : $"image {key.Ordinal} in {key.PartUri} changed but was not approved");
            }

            imagesChecked++;
        }

        return new DocxValidationResult(problems.Count == 0, problems, compared, imagesChecked);

        void Note(string problem)
        {
            if (problems.Count < MaxProblems)
            {
                problems.Add(problem);
            }
        }
    }

    private static ZipArchive? OpenOrReport(string path, List<string> problems)
    {
        try
        {
            return ZipFile.OpenRead(path);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException)
        {
            problems.Add($"the output is not a valid package: {ex.Message}");
            return null;
        }
    }

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    /// <summary>SHA-256 of image parts, computed once per part by streaming the zip entry.</summary>
    private sealed class ImageHashes(string docxPath) : IDisposable
    {
        private readonly ZipArchive _zip = ZipFile.OpenRead(docxPath);
        private readonly Dictionary<string, string> _cache = new(StringComparer.OrdinalIgnoreCase);

        public string Get(string partUri)
        {
            if (_cache.TryGetValue(partUri, out var hash))
            {
                return hash;
            }

            var entry = DocxScanner.FindEntry(_zip, partUri);
            if (entry is null)
            {
                return _cache[partUri] = "missing";
            }

            using var stream = entry.Open();
            return _cache[partUri] = Convert.ToHexString(SHA256.HashData(stream));
        }

        public void Dispose() => _zip.Dispose();
    }
}
