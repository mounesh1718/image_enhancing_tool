using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Linq;

namespace ImageQualityEnhancer.Services.Docx;

/// <summary>Replace the image shown at (PartUri, Ordinal), currently <paramref name="ImageUri"/>, with an enhanced file.</summary>
public sealed record DocxReplacement(string PartUri, int Ordinal, string ImageUri, string EnhancedPath);

public sealed record DocxReplacementFailure(DocxReplacement Replacement, string Code, string Message);

public sealed record DocxExportProgress(string Stage, int Done, int Total);

public sealed record DocxExportResult(
    IReadOnlyList<DocxReplacement> Applied,
    IReadOnlyList<DocxReplacementFailure> Failed,
    DocxValidationResult Validation,
    int ImagesReplacedInPlace,
    int ImagesAdded,
    long FileSize);

/// <summary>
/// Writes a copy of a .docx with approved images replaced, in one streaming pass over the zip.
///
/// The Open XML SDK's editable mode loads the whole package into memory on save (measured: 324 MB to change one
/// image in a 297 MB document), so the output is built entry by entry instead:
/// - untouched entries are copied through;
/// - an image whose every occurrence is approved (and keeps its format) has its bytes swapped in place;
/// - otherwise the enhanced image becomes a new part and only the approved drawings are pointed at it,
///   so unapproved occurrences of a shared image stay exactly as they were;
/// - only the parts holding those drawings, their .rels and [Content_Types].xml are rewritten, node by node.
/// The result is validated against the original before it is moved to <c>output</c>.
/// </summary>
public static class DocxExporter
{
    private const string RelationshipsNs = "http://schemas.openxmlformats.org/package/2006/relationships";
    private const string ContentTypesNs = "http://schemas.openxmlformats.org/package/2006/content-types";
    private const string ImageRelType = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image";

    private sealed class Plan
    {
        public Dictionary<string, string> InPlace { get; } = new(StringComparer.OrdinalIgnoreCase); // entry -> enhanced file
        public Dictionary<string, NewMedia> NewMediaByFile { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, Dictionary<int, string>> Redirects { get; } = new(StringComparer.OrdinalIgnoreCase); // part -> ordinal -> rel id
        public Dictionary<string, List<(string Id, string Target)>> NewRelationships { get; } = new(StringComparer.OrdinalIgnoreCase); // rels entry -> rels
        public Dictionary<string, string> NewDefaults { get; } = new(StringComparer.OrdinalIgnoreCase); // extension -> content type
    }

    private sealed record NewMedia(string EntryName, string EnhancedPath, string Extension, string ContentType);

    public static DocxExportResult Export(
        string sourcePath,
        string outputPath,
        IReadOnlyList<DocxReplacement> replacements,
        IProgress<DocxExportProgress>? progress = null,
        CancellationToken ct = default)
    {
        if (string.Equals(Path.GetFullPath(sourcePath), Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase))
        {
            throw new DocxException("bad_request", "the output must be a different file from the original");
        }

        progress?.Report(new DocxExportProgress("scan", 0, 1));
        var scan = DocxScanner.Scan(sourcePath, ct: ct);
        var occurrences = scan.Occurrences.ToDictionary(o => (o.PartUri, o.Ref.Ordinal));
        var imageTypes = scan.Images.ToDictionary(i => i.Uri, i => i.ContentType, StringComparer.OrdinalIgnoreCase);

        // Check every request against the current document; invalid ones fail alone.
        var applied = new List<DocxReplacement>();
        var failed = new List<DocxReplacementFailure>();
        foreach (var r in replacements)
        {
            var problem = Check(r, occurrences);
            if (problem is null)
            {
                applied.Add(r);
            }
            else
            {
                failed.Add(new DocxReplacementFailure(r, problem.Value.Code, problem.Value.Message));
            }
        }

        using var source = OpenZip(sourcePath);
        var relsBySourcePart = ReadAllRelationships(source);
        var plan = BuildPlan(scan, applied, imageTypes, relsBySourcePart, source);

        var tmp = outputPath + ".tmp";
        try
        {
            WriteOutput(source, tmp, plan, progress, ct);
            source.Dispose();

            progress?.Report(new DocxExportProgress("validate", 0, 1));
            var appliedMap = applied.ToDictionary(r => (r.PartUri, r.Ordinal), r => r.EnhancedPath);
            var changedEntries = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            changedEntries.UnionWith(plan.InPlace.Keys);
            changedEntries.UnionWith(plan.Redirects.Keys.Select(p => p.TrimStart('/')));
            changedEntries.UnionWith(plan.NewRelationships.Keys);
            if (plan.NewDefaults.Count > 0)
            {
                changedEntries.Add("[Content_Types].xml");
            }

            var validation = DocxValidator.Validate(sourcePath, tmp, scan, appliedMap, changedEntries, ct);
            if (!validation.Ok)
            {
                throw new DocxException("validation_failed",
                    "the new document did not pass validation: " + string.Join("; ", validation.Problems.Take(5)));
            }

            File.Move(tmp, outputPath, overwrite: true);
            return new DocxExportResult(applied, failed, validation, plan.InPlace.Count, plan.NewMediaByFile.Count,
                new FileInfo(outputPath).Length);
        }
        catch (IOException ex) when (IsDiskFull(ex))
        {
            throw new DocxException("disk_full", "there is not enough disk space to write the document", ex);
        }
        finally
        {
            if (File.Exists(tmp))
            {
                File.Delete(tmp);
            }
        }
    }

    private static (string Code, string Message)? Check(DocxReplacement r, Dictionary<(string, int), DocxImageOccurrence> occurrences)
    {
        if (!occurrences.TryGetValue((r.PartUri, r.Ordinal), out var occurrence))
        {
            return ("missing_occurrence", "the image is no longer at this position in the document");
        }

        if (!string.Equals(occurrence.ImageUri, r.ImageUri, StringComparison.OrdinalIgnoreCase))
        {
            return ("changed_occurrence", "a different image is now shown at this position");
        }

        if (occurrence.SkipReason is not null)
        {
            return ("not_replaceable", $"this image cannot be replaced ({occurrence.SkipReason})");
        }

        if (!File.Exists(r.EnhancedPath))
        {
            return ("missing_enhanced_file", "the enhanced image file is missing");
        }

        return ImageSignature.Detect(r.EnhancedPath) is null
            ? ("unsupported_format", "the enhanced file is not a supported image format")
            : null;
    }

    private static Plan BuildPlan(
        DocxScanResult scan,
        List<DocxReplacement> applied,
        Dictionary<string, string> imageTypes,
        Dictionary<string, List<(string RelsEntry, string Id, string TargetUri)>> relsBySourcePart,
        ZipArchive source)
    {
        var plan = new Plan();
        var scannedParts = new HashSet<string>(scan.Occurrences.Select(o => o.PartUri), StringComparer.OrdinalIgnoreCase);
        var existingNames = new HashSet<string>(source.Entries.Select(e => e.FullName), StringComparer.OrdinalIgnoreCase);
        var defaults = ReadDefaultExtensions(source);
        var mediaCounter = 0;
        var relCounter = 0;

        foreach (var group in applied.GroupBy(r => r.ImageUri, StringComparer.OrdinalIgnoreCase))
        {
            var imageUri = group.Key;
            var all = scan.Occurrences.Where(o => string.Equals(o.ImageUri, imageUri, StringComparison.OrdinalIgnoreCase)).ToList();
            var files = group.Select(r => r.EnhancedPath).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
            var originalType = imageTypes[imageUri];
            var newType = ImageSignature.Detect(files[0])!.Value.ContentType;

            // Also referenced from parts we do not scan (charts, SmartArt, glossary)? Then never touch it in place.
            var referencedElsewhere = relsBySourcePart
                .Where(kv => !scannedParts.Contains(kv.Key))
                .Any(kv => kv.Value.Any(rel => SamePart(rel.TargetUri, imageUri)));

            var everyOccurrenceApproved = group.Count() == all.Count;
            if (everyOccurrenceApproved && files.Count == 1 && !referencedElsewhere
                && string.Equals(NormalizeType(originalType), newType, StringComparison.OrdinalIgnoreCase))
            {
                plan.InPlace[imageUri.TrimStart('/')] = files[0];
                continue;
            }

            foreach (var r in group)
            {
                if (!plan.NewMediaByFile.TryGetValue(r.EnhancedPath, out var media))
                {
                    var signature = ImageSignature.Detect(r.EnhancedPath)!.Value;
                    string name;
                    do
                    {
                        name = $"word/media/iqe_image{++mediaCounter}{signature.Extension}";
                    }
                    while (existingNames.Contains(name));

                    media = new NewMedia(name, r.EnhancedPath, signature.Extension.TrimStart('.'), signature.ContentType);
                    plan.NewMediaByFile[r.EnhancedPath] = media;
                    if (!defaults.Contains(media.Extension))
                    {
                        plan.NewDefaults[media.Extension] = media.ContentType;
                    }
                }

                var relsEntry = RelsEntryFor(r.PartUri);
                var rels = plan.NewRelationships.TryGetValue(relsEntry, out var list) ? list : plan.NewRelationships[relsEntry] = [];
                var target = RelativeTarget(r.PartUri, "/" + media.EntryName);
                var existing = rels.FirstOrDefault(x => x.Target == target);
                var relId = existing.Id;
                if (relId is null)
                {
                    var used = relsBySourcePart.TryGetValue(r.PartUri, out var current)
                        ? current.Select(x => x.Id).ToHashSet(StringComparer.Ordinal)
                        : [];
                    do
                    {
                        relId = $"rIdIqe{++relCounter}";
                    }
                    while (used.Contains(relId));

                    rels.Add((relId, target));
                }

                var redirects = plan.Redirects.TryGetValue(r.PartUri, out var map) ? map : plan.Redirects[r.PartUri] = [];
                redirects[r.Ordinal] = relId;
            }
        }

        return plan;
    }

    private static void WriteOutput(ZipArchive source, string tmp, Plan plan, IProgress<DocxExportProgress>? progress, CancellationToken ct)
    {
        var redirectEntries = plan.Redirects.ToDictionary(kv => kv.Key.TrimStart('/'), kv => kv.Value, StringComparer.OrdinalIgnoreCase);
        var total = source.Entries.Count + plan.NewMediaByFile.Count;
        var done = 0;

        using var output = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        using var zip = new ZipArchive(output, ZipArchiveMode.Create);

        foreach (var entry in source.Entries)
        {
            ct.ThrowIfCancellationRequested();
            var name = entry.FullName;
            if (name.EndsWith('/'))
            {
                zip.CreateEntry(name);
                continue;
            }

            var target = zip.CreateEntry(name, LevelFor(name));
            target.LastWriteTime = entry.LastWriteTime;
            using (var to = target.Open())
            {
                if (name == "[Content_Types].xml" && plan.NewDefaults.Count > 0)
                {
                    using var from = entry.Open();
                    WriteContentTypes(from, to, plan.NewDefaults);
                }
                else if (plan.NewRelationships.TryGetValue(name, out var rels))
                {
                    using var from = entry.Open();
                    WriteRelationships(from, to, rels);
                }
                else if (redirectEntries.TryGetValue(name, out var redirects))
                {
                    using var from = entry.Open();
                    RewritePart(from, to, redirects);
                }
                else if (plan.InPlace.TryGetValue(name, out var enhanced))
                {
                    using var from = File.OpenRead(enhanced);
                    from.CopyTo(to);
                }
                else
                {
                    using var from = entry.Open();
                    from.CopyTo(to);
                }
            }

            progress?.Report(new DocxExportProgress("write", ++done, total));
        }

        foreach (var media in plan.NewMediaByFile.Values)
        {
            ct.ThrowIfCancellationRequested();
            var entry = zip.CreateEntry(media.EntryName, CompressionLevel.Fastest);
            using (var to = entry.Open())
            using (var from = File.OpenRead(media.EnhancedPath))
            {
                from.CopyTo(to);
            }

            progress?.Report(new DocxExportProgress("write", ++done, total));
        }
    }

    private static void RewritePart(Stream from, Stream to, Dictionary<int, string> redirects)
    {
        using var reader = XmlReader.Create(from, DocxScanner.ReaderSettings);
        using var writer = XmlWriter.Create(to, new XmlWriterSettings
        {
            Encoding = new UTF8Encoding(false),
            Indent = false,
            NewLineHandling = NewLineHandling.None,
            CloseOutput = false,
        });
        DocxImageWalker.Walk(reader, writer, image => redirects.GetValueOrDefault(image.Ordinal));
    }

    private static void WriteRelationships(Stream from, Stream to, List<(string Id, string Target)> additions)
    {
        var doc = XDocument.Load(from);
        XNamespace ns = RelationshipsNs;
        foreach (var (id, target) in additions)
        {
            doc.Root!.Add(new XElement(ns + "Relationship",
                new XAttribute("Id", id), new XAttribute("Type", ImageRelType), new XAttribute("Target", target)));
        }

        SaveXml(doc, to);
    }

    private static void WriteContentTypes(Stream from, Stream to, Dictionary<string, string> defaults)
    {
        var doc = XDocument.Load(from);
        XNamespace ns = ContentTypesNs;
        foreach (var (extension, contentType) in defaults)
        {
            doc.Root!.AddFirst(new XElement(ns + "Default",
                new XAttribute("Extension", extension), new XAttribute("ContentType", contentType)));
        }

        SaveXml(doc, to);
    }

    private static void SaveXml(XDocument doc, Stream to)
    {
        using var writer = XmlWriter.Create(to, new XmlWriterSettings { Encoding = new UTF8Encoding(false), CloseOutput = false });
        doc.Save(writer);
    }

    private static HashSet<string> ReadDefaultExtensions(ZipArchive zip)
    {
        var entry = zip.GetEntry("[Content_Types].xml") ?? throw new DocxException("corrupt_docx", "the document has no content types");
        using var stream = entry.Open();
        XNamespace ns = ContentTypesNs;
        return XDocument.Load(stream).Root!.Elements(ns + "Default")
            .Select(e => (string?)e.Attribute("Extension") ?? "")
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>All internal relationships in the package, keyed by source part URI ("/" for the package itself).</summary>
    private static Dictionary<string, List<(string RelsEntry, string Id, string TargetUri)>> ReadAllRelationships(ZipArchive zip)
    {
        var result = new Dictionary<string, List<(string, string, string)>>(StringComparer.OrdinalIgnoreCase);
        XNamespace ns = RelationshipsNs;
        foreach (var entry in zip.Entries.Where(e => e.FullName.EndsWith(".rels", StringComparison.OrdinalIgnoreCase)))
        {
            var sourcePart = SourcePartOf(entry.FullName);
            if (sourcePart is null)
            {
                continue;
            }

            XDocument doc;
            using (var stream = entry.Open())
            {
                doc = XDocument.Load(stream);
            }

            var list = new List<(string, string, string)>();
            foreach (var rel in doc.Root!.Elements(ns + "Relationship"))
            {
                if (string.Equals((string?)rel.Attribute("TargetMode"), "External", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var id = (string?)rel.Attribute("Id") ?? "";
                var target = (string?)rel.Attribute("Target") ?? "";
                list.Add((entry.FullName, id, ResolveTarget(sourcePart, target)));
            }

            result[sourcePart] = list;
        }

        return result;
    }

    /// <summary>"word/_rels/document.xml.rels" -> "/word/document.xml"; "_rels/.rels" -> "/".</summary>
    private static string? SourcePartOf(string relsEntry)
    {
        var slash = relsEntry.LastIndexOf("_rels/", StringComparison.Ordinal);
        if (slash < 0)
        {
            return null;
        }

        var folder = relsEntry[..slash];
        var file = relsEntry[(slash + "_rels/".Length)..];
        file = file.EndsWith(".rels", StringComparison.OrdinalIgnoreCase) ? file[..^5] : file;
        return "/" + folder + file;
    }

    internal static string RelsEntryFor(string partUri)
    {
        var trimmed = partUri.TrimStart('/');
        var slash = trimmed.LastIndexOf('/');
        return slash < 0 ? $"_rels/{trimmed}.rels" : $"{trimmed[..(slash + 1)]}_rels/{trimmed[(slash + 1)..]}.rels";
    }

    private static readonly Uri PackageRoot = new("http://package/");

    private static string ResolveTarget(string sourcePart, string target)
    {
        var baseUri = new Uri(PackageRoot, sourcePart.TrimStart('/'));
        return new Uri(baseUri, target).AbsolutePath;
    }

    private static string RelativeTarget(string sourcePart, string targetPart)
    {
        var from = new Uri(PackageRoot, sourcePart.TrimStart('/'));
        var to = new Uri(PackageRoot, targetPart.TrimStart('/'));
        return from.MakeRelativeUri(to).OriginalString;
    }

    private static bool SamePart(string a, string b) =>
        string.Equals(Uri.UnescapeDataString(a), Uri.UnescapeDataString(b), StringComparison.OrdinalIgnoreCase);

    private static string NormalizeType(string contentType) => contentType.ToLowerInvariant() switch
    {
        "image/x-png" => "image/png",
        "image/jpg" or "image/pjpeg" => "image/jpeg",
        "image/x-bmp" => "image/bmp",
        var other => other,
    };

    private static CompressionLevel LevelFor(string name) =>
        name.EndsWith(".xml", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".rels", StringComparison.OrdinalIgnoreCase)
            ? CompressionLevel.Optimal
            : CompressionLevel.Fastest;

    private static ZipArchive OpenZip(string path)
    {
        try
        {
            return ZipFile.OpenRead(path);
        }
        catch (InvalidDataException ex)
        {
            throw new DocxException("corrupt_docx", "the Word document is damaged", ex);
        }
    }

    private static bool IsDiskFull(IOException ex) =>
        (ex.HResult & 0xFFFF) is 0x70 or 0x27; // ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL
}

/// <summary>Identifies an image file from its first bytes.</summary>
internal static class ImageSignature
{
    public static (string Extension, string ContentType)? Detect(string path)
    {
        Span<byte> head = stackalloc byte[12];
        int read;
        using (var stream = File.OpenRead(path))
        {
            read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        }

        head = head[..read];
        if (head.StartsWith(new byte[] { 0x89, 0x50, 0x4E, 0x47 })) return (".png", "image/png");
        if (head.StartsWith(new byte[] { 0xFF, 0xD8, 0xFF })) return (".jpeg", "image/jpeg");
        if (head.StartsWith("GIF8"u8)) return (".gif", "image/gif");
        if (head.StartsWith("BM"u8)) return (".bmp", "image/bmp");
        if (head.StartsWith("II*\0"u8) || head.StartsWith("MM\0*"u8)) return (".tiff", "image/tiff");
        if (read >= 12 && head[..4].SequenceEqual("RIFF"u8) && head[8..12].SequenceEqual("WEBP"u8)) return (".webp", "image/webp");
        return null;
    }
}
