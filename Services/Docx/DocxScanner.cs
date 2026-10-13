using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using System.Xml;
using DocumentFormat.OpenXml.Packaging;

namespace ImageQualityEnhancer.Services.Docx;

/// <summary>A document-level problem (corrupt file, password protection, export failure).</summary>
public sealed class DocxException(string code, string message, Exception? inner = null) : Exception(message, inner)
{
    public string Code { get; } = code;
}

/// <summary>One place where an image is shown in the document.</summary>
public sealed record DocxImageOccurrence
{
    /// <summary>Part holding the drawing, e.g. /word/document.xml or /word/header1.xml.</summary>
    public required string PartUri { get; init; }

    public required DocxImageRef Ref { get; init; }

    /// <summary>The image part shown, e.g. /word/media/image3.png (null when missing or external).</summary>
    public string? ImageUri { get; init; }

    /// <summary>Why this occurrence must stay unchanged; null when it may be enhanced.</summary>
    public string? SkipReason { get; init; }
}

/// <summary>One stored image file, possibly shown in several places.</summary>
public sealed record DocxImageResource
{
    public required string Uri { get; init; }
    public required string ContentType { get; init; }
    public long Length { get; init; }
    public int ReferenceCount { get; init; }

    /// <summary>Worker output format that can be stored back under the same content type, or null if not enhanceable.</summary>
    public string? OutputFormat { get; init; }

    public string? SkipReason { get; init; }
}

public sealed record DocxScanResult(
    IReadOnlyList<DocxImageOccurrence> Occurrences,
    IReadOnlyList<DocxImageResource> Images,
    IReadOnlyDictionary<string, long> PartElementCounts,
    IReadOnlyDictionary<string, string> PartContentHashes);

/// <summary>Finds every image reference in a .docx, reading parts as streams (no document DOM is built).</summary>
public static class DocxScanner
{
    private static readonly Dictionary<string, string?> FormatsByContentType = new(StringComparer.OrdinalIgnoreCase)
    {
        ["image/png"] = "png",
        ["image/x-png"] = "png",
        ["image/jpeg"] = "jpeg",
        ["image/jpg"] = "jpeg",
        ["image/pjpeg"] = "jpeg",
        ["image/bmp"] = "bmp",
        ["image/x-bmp"] = "bmp",
        ["image/tiff"] = "tiff",
        // GIF is limited to 256 colours: an enhanced GIF is stored as a new PNG part instead.
        ["image/gif"] = "png",
        ["image/webp"] = "webp",
    };

    private static readonly HashSet<string> VectorContentTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "image/x-emf", "image/emf", "image/x-wmf", "image/wmf", "image/svg+xml", "image/x-pict",
    };

    public static DocxScanResult Scan(string path, IProgress<int>? partsDone = null, CancellationToken ct = default)
    {
        EnsureOpenable(path);
        using var doc = OpenReadOnly(path);
        var main = doc.MainDocumentPart ?? throw new DocxException("corrupt_docx", "the document has no main body");

        var occurrences = new List<DocxImageOccurrence>();
        var elementCounts = new Dictionary<string, long>();
        var contentHashes = new Dictionary<string, string>();
        var resources = new Dictionary<string, (ImagePart Part, int Count)>();
        var done = 0;

        foreach (var part in ContentParts(main))
        {
            ct.ThrowIfCancellationRequested();
            var partUri = part.Uri.OriginalString;
            DocxImageWalker.Result result;
            try
            {
                using var stream = part.GetStream(FileMode.Open, FileAccess.Read);
                using var reader = XmlReader.Create(stream, ReaderSettings);
                result = DocxImageWalker.Scan(reader);
            }
            catch (XmlException ex)
            {
                throw new DocxException("corrupt_docx", $"{partUri} is damaged: {ex.Message}", ex);
            }

            elementCounts[partUri] = result.ElementCount;
            contentHashes[partUri] = result.ContentHash;
            foreach (var image in result.Images)
            {
                occurrences.Add(Resolve(part, partUri, image, resources));
            }

            partsDone?.Report(++done);
        }

        var images = resources.Values
            .Select(r => Describe(r.Part, r.Count))
            .OrderBy(r => r.Uri, StringComparer.Ordinal)
            .ToList();
        return new DocxScanResult(occurrences, images, elementCounts, contentHashes);
    }

    /// <summary>Copies one stored image to <paramref name="targetPath"/> exactly as stored.</summary>
    public static void ExtractImage(string docxPath, string imageUri, string targetPath)
    {
        using var zip = ZipFile.OpenRead(docxPath);
        var entry = FindEntry(zip, imageUri) ?? throw new DocxException("missing_image", $"{imageUri} is not in the document");
        var tmp = targetPath + ".tmp";
        try
        {
            using (var source = entry.Open())
            using (var target = File.Create(tmp))
            {
                source.CopyTo(target);
            }

            File.Move(tmp, targetPath, overwrite: true);
        }
        catch
        {
            File.Delete(tmp);
            throw;
        }
    }

    internal static readonly XmlReaderSettings ReaderSettings = new()
    {
        DtdProcessing = DtdProcessing.Prohibit,
        XmlResolver = null,
        IgnoreWhitespace = false,
        CloseInput = false,
    };

    /// <summary>Parts that can show images: body, headers, footers, footnotes, endnotes, comments.</summary>
    internal static IEnumerable<OpenXmlPart> ContentParts(MainDocumentPart main)
    {
        yield return main;
        foreach (var part in main.HeaderParts) yield return part;
        foreach (var part in main.FooterParts) yield return part;
        if (main.FootnotesPart is not null) yield return main.FootnotesPart;
        if (main.EndnotesPart is not null) yield return main.EndnotesPart;
        if (main.WordprocessingCommentsPart is not null) yield return main.WordprocessingCommentsPart;
    }

    internal static WordprocessingDocument OpenReadOnly(string path)
    {
        try
        {
            return WordprocessingDocument.Open(path, false);
        }
        catch (Exception ex) when (ex is OpenXmlPackageException or InvalidDataException or FileFormatException
                                       or IOException or ArgumentException or InvalidOperationException)
        {
            throw new DocxException("corrupt_docx", "the Word document is damaged or is not a .docx file", ex);
        }
    }

    /// <summary>Recognises files that cannot be opened as .docx and explains why.</summary>
    internal static void EnsureOpenable(string path)
    {
        if (!File.Exists(path))
        {
            throw new DocxException("file_not_found", "the Word document does not exist");
        }

        var head = new byte[64 * 1024];
        int read;
        using (var stream = File.OpenRead(path))
        {
            read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        }

        if (read >= 4 && head[0] == 0x50 && head[1] == 0x4B && head[2] == 0x03 && head[3] == 0x04)
        {
            return; // zip container
        }

        if (read >= 8 && head.AsSpan(0, 8).SequenceEqual(new byte[] { 0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1 }))
        {
            // Encrypted .docx files are stored inside an OLE compound file with an "EncryptedPackage" stream.
            var marker = Encoding.Unicode.GetBytes("EncryptedPackage");
            if (head.AsSpan(0, read).IndexOf(marker) >= 0)
            {
                throw new DocxException("password_protected", "the Word document is password-protected; remove the password in Word first");
            }

            throw new DocxException("legacy_doc", "this is an old .doc file; save it as .docx in Word first");
        }

        throw new DocxException("corrupt_docx", "the file is not a Word .docx document");
    }

    internal static ZipArchiveEntry? FindEntry(ZipArchive zip, string partUri)
    {
        var name = partUri.TrimStart('/');
        return zip.GetEntry(name)
               ?? zip.Entries.FirstOrDefault(e => string.Equals(Uri.UnescapeDataString(e.FullName), Uri.UnescapeDataString(name), StringComparison.OrdinalIgnoreCase));
    }

    private static DocxImageOccurrence Resolve(OpenXmlPart part, string partUri, DocxImageRef image,
        Dictionary<string, (ImagePart Part, int Count)> resources)
    {
        if (image.Kind == DocxImageKind.Linked)
        {
            return new DocxImageOccurrence { PartUri = partUri, Ref = image, SkipReason = "linked_image" };
        }

        if (image.RelId is null || !part.TryGetPartById(image.RelId, out var target) || target is not ImagePart imagePart)
        {
            return new DocxImageOccurrence { PartUri = partUri, Ref = image, SkipReason = "missing_image" };
        }

        var uri = imagePart.Uri.OriginalString;
        resources[uri] = resources.TryGetValue(uri, out var existing) ? (imagePart, existing.Count + 1) : (imagePart, 1);

        string? skip = null;
        if (image.Kind == DocxImageKind.OlePreview)
        {
            skip = "ole_preview"; // the embedded object's program redraws this preview
        }
        else if (image.HasSvg)
        {
            skip = "svg_fallback"; // Word shows the SVG; this PNG is only for old versions
        }
        else if (VectorContentTypes.Contains(imagePart.ContentType))
        {
            skip = "vector";
        }
        else if (!FormatsByContentType.ContainsKey(imagePart.ContentType))
        {
            skip = "unsupported_format";
        }

        return new DocxImageOccurrence { PartUri = partUri, Ref = image, ImageUri = uri, SkipReason = skip };
    }

    private static DocxImageResource Describe(ImagePart part, int count)
    {
        long length;
        using (var stream = part.GetStream(FileMode.Open, FileAccess.Read))
        {
            length = stream.CanSeek ? stream.Length : -1;
        }

        var contentType = part.ContentType;
        FormatsByContentType.TryGetValue(contentType, out var format);
        return new DocxImageResource
        {
            Uri = part.Uri.OriginalString,
            ContentType = contentType,
            Length = length,
            ReferenceCount = count,
            OutputFormat = format,
            SkipReason = format is not null ? null : VectorContentTypes.Contains(contentType) ? "vector" : "unsupported_format",
        };
    }
}
