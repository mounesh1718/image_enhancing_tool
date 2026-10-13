using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;

namespace ImageQualityEnhancer.Services.Docx;

public enum DocxImageKind
{
    /// <summary>DrawingML picture or picture fill (a:blip r:embed).</summary>
    Picture,

    /// <summary>Legacy VML image (v:imagedata) outside an OLE object.</summary>
    Vml,

    /// <summary>Preview image of an embedded OLE object (e.g. a ChemDraw structure). The OLE server redraws it.</summary>
    OlePreview,

    /// <summary>Picture stored outside the document (a:blip r:link).</summary>
    Linked,
}

/// <summary>One image reference found in a document part, in document order.</summary>
public sealed class DocxImageRef
{
    /// <summary>Position among the part's image references, counting only the primary (non-fallback) markup.</summary>
    public int Ordinal { get; init; }

    public required string? RelId { get; init; }
    public required DocxImageKind Kind { get; init; }
    public bool InTable { get; init; }
    public bool Anchored { get; init; }

    /// <summary>wp:docPr id/name/descr of the containing drawing (null for VML).</summary>
    public string? DrawingId { get; init; }
    public string? Name { get; init; }
    public string? Description { get; init; }

    /// <summary>Displayed size in EMU (914400 per inch) from wp:extent, or from the VML style for VML images.</summary>
    public long? ExtentCx { get; init; }
    public long? ExtentCy { get; init; }

    /// <summary>a:srcRect cropping (left, top, right, bottom) in 1/1000 of a percent, if any.</summary>
    public int[]? Crop { get; set; }

    /// <summary>The picture is an SVG; this reference is only its PNG fallback for old Word versions.</summary>
    public bool HasSvg { get; set; }
}

/// <summary>
/// Streams through one WordprocessingML part (document, header, footer, notes, comments) and reports every image
/// reference, optionally copying the XML to a writer while redirecting selected references to new relationship ids.
///
/// Scanning and rewriting use this same walker, so an image's ordinal means the same thing in both. Content
/// inside mc:Fallback (the copy Word keeps for older versions) is not counted; when a reference is redirected,
/// matching ids in the AlternateContent's fallback are redirected too.
/// </summary>
public sealed class DocxImageWalker
{
    private const string W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
    private const string WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing";
    private const string A = "http://schemas.openxmlformats.org/drawingml/2006/main";
    private const string R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    private const string V = "urn:schemas-microsoft-com:vml";
    private const string O = "urn:schemas-microsoft-com:office:office";
    private const string MC = "http://schemas.openxmlformats.org/markup-compatibility/2006";
    private const string ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main";

    private static readonly Regex VmlSize = new(@"(width|height)\s*:\s*([\d.]+)\s*(pt|in|cm|mm|px)?", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    /// <param name="ContentHash">SHA-256 of all text in the part, to prove a rewrite left the text untouched.</param>
    public sealed record Result(IReadOnlyList<DocxImageRef> Images, long ElementCount, string ContentHash);

    private sealed class Drawing
    {
        public bool Anchored;
        public long? Cx, Cy;
        public string? Id, Name, Description;
    }

    /// <summary>Scans a part without writing anything.</summary>
    public static Result Scan(XmlReader reader) => Walk(reader, null, null);

    /// <summary>
    /// Walks the part. When <paramref name="writer"/> is set, every node is copied to it unchanged, except that
    /// the relationship attribute of an image reference is replaced when <paramref name="redirect"/> returns a new id.
    /// </summary>
    public static Result Walk(XmlReader reader, XmlWriter? writer, Func<DocxImageRef, string?>? redirect)
    {
        var images = new List<DocxImageRef>();
        var drawings = new Stack<Drawing>();
        var shapeStyles = new Stack<string?>();
        // One map per open mc:AlternateContent: ids redirected in its Choice, applied again in its Fallback.
        var alternateMaps = new Stack<Dictionary<string, string>>();
        int tableDepth = 0, fallbackDepth = 0, objectDepth = 0, ordinal = 0;
        long elements = 0;
        DocxImageRef? currentBlip = null;
        DocxImageRef? fillBlip = null;
        var ends = new Stack<Action?>();
        using var textHash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        while (reader.Read())
        {
            switch (reader.NodeType)
            {
                case XmlNodeType.Element:
                {
                    elements++;
                    var ns = reader.NamespaceURI;
                    var local = reader.LocalName;
                    var empty = reader.IsEmptyElement;
                    Action? onEnd = null;
                    string? redirectFrom = null, redirectTo = null, redirectAttrNs = null, redirectAttrLocal = null;

                    if (ns == W && local == "tc")
                    {
                        tableDepth++;
                        onEnd = () => tableDepth--;
                    }
                    else if (ns == W && local == "object")
                    {
                        objectDepth++;
                        onEnd = () => objectDepth--;
                    }
                    else if (ns == MC && local == "AlternateContent")
                    {
                        alternateMaps.Push(new Dictionary<string, string>());
                        onEnd = () => alternateMaps.Pop();
                    }
                    else if (ns == MC && local == "Fallback")
                    {
                        fallbackDepth++;
                        onEnd = () => fallbackDepth--;
                    }
                    else if (ns == WP && (local == "inline" || local == "anchor"))
                    {
                        drawings.Push(new Drawing { Anchored = local == "anchor" });
                        onEnd = () => drawings.Pop();
                    }
                    else if (ns == WP && local == "extent" && drawings.Count > 0)
                    {
                        drawings.Peek().Cx = ParseLong(reader.GetAttribute("cx"));
                        drawings.Peek().Cy = ParseLong(reader.GetAttribute("cy"));
                    }
                    else if (ns == WP && local == "docPr" && drawings.Count > 0)
                    {
                        var d = drawings.Peek();
                        d.Id = reader.GetAttribute("id");
                        d.Name = reader.GetAttribute("name");
                        d.Description = reader.GetAttribute("descr");
                    }
                    else if (ns == V && (local == "shape" || local == "rect" || local == "image"))
                    {
                        shapeStyles.Push(reader.GetAttribute("style"));
                        onEnd = () => shapeStyles.Pop();
                    }
                    else if (ns == A && local == "blipFill")
                    {
                        fillBlip = null;
                    }
                    else if (ns == A && local == "srcRect" && fillBlip is not null)
                    {
                        var crop = new[] { "l", "t", "r", "b" }.Select(n => (int)(ParseLong(reader.GetAttribute(n)) ?? 0)).ToArray();
                        if (crop.Any(c => c != 0))
                        {
                            fillBlip.Crop = crop;
                        }
                    }
                    else if (ns == ASVG && local == "svgBlip" && currentBlip is not null)
                    {
                        currentBlip.HasSvg = true;
                    }
                    else if (ns == A && local == "blip")
                    {
                        var embed = reader.GetAttribute("embed", R);
                        var link = reader.GetAttribute("link", R);
                        var drawing = drawings.Count > 0 ? drawings.Peek() : null;
                        var image = new DocxImageRef
                        {
                            Ordinal = fallbackDepth > 0 ? -1 : ordinal,
                            RelId = embed ?? link,
                            Kind = embed is null && link is not null ? DocxImageKind.Linked : DocxImageKind.Picture,
                            InTable = tableDepth > 0,
                            Anchored = drawing?.Anchored ?? false,
                            DrawingId = drawing?.Id,
                            Name = drawing?.Name,
                            Description = drawing?.Description,
                            ExtentCx = drawing?.Cx,
                            ExtentCy = drawing?.Cy,
                        };
                        fillBlip = image;
                        if (!empty)
                        {
                            currentBlip = image;
                            onEnd = () => currentBlip = null;
                        }

                        Record(image, embed is not null ? "embed" : "link", R);
                    }
                    else if (ns == V && local == "imagedata")
                    {
                        var id = reader.GetAttribute("id", R);
                        var attrNs = R;
                        if (id is null)
                        {
                            id = reader.GetAttribute("relid", O);
                            attrNs = O;
                        }

                        var (cx, cy) = ParseVmlSize(shapeStyles.Count > 0 ? shapeStyles.Peek() : null);
                        var image = new DocxImageRef
                        {
                            Ordinal = fallbackDepth > 0 ? -1 : ordinal,
                            RelId = id,
                            Kind = objectDepth > 0 ? DocxImageKind.OlePreview : DocxImageKind.Vml,
                            InTable = tableDepth > 0,
                            ExtentCx = cx,
                            ExtentCy = cy,
                        };
                        Record(image, attrNs == R ? "id" : "relid", attrNs);
                    }

                    // Fallback markup repeats the Choice's ids: follow any redirect made there.
                    if (writer is not null && fallbackDepth > 0 && alternateMaps.Count > 0 && alternateMaps.Peek().Count > 0
                        && redirectFrom is null && reader.HasAttributes)
                    {
                        CopyElement(reader, writer, alternateMaps.Peek(), null, null, null, empty);
                    }
                    else if (writer is not null)
                    {
                        CopyElement(reader, writer, null, redirectAttrNs, redirectAttrLocal, redirectTo, empty);
                    }

                    if (!empty)
                    {
                        ends.Push(onEnd);
                    }
                    else
                    {
                        onEnd?.Invoke();
                    }

                    void Record(DocxImageRef image, string attrLocal, string attrNs)
                    {
                        if (image.Ordinal >= 0)
                        {
                            images.Add(image);
                            ordinal++;
                            var target = redirect?.Invoke(image);
                            if (target is not null && image.RelId is not null)
                            {
                                redirectFrom = image.RelId;
                                redirectTo = target;
                                redirectAttrNs = attrNs;
                                redirectAttrLocal = attrLocal;
                                if (alternateMaps.Count > 0)
                                {
                                    alternateMaps.Peek()[image.RelId] = target;
                                }
                            }
                        }
                    }

                    break;
                }

                case XmlNodeType.EndElement:
                    ends.Pop()?.Invoke();
                    writer?.WriteFullEndElement();
                    break;

                case XmlNodeType.Text:
                    textHash.AppendData(Encoding.UTF8.GetBytes(reader.Value));
                    writer?.WriteString(reader.Value);
                    break;

                case XmlNodeType.Whitespace:
                case XmlNodeType.SignificantWhitespace:
                    textHash.AppendData(Encoding.UTF8.GetBytes(reader.Value));
                    writer?.WriteWhitespace(reader.Value);
                    break;

                case XmlNodeType.CDATA:
                    textHash.AppendData(Encoding.UTF8.GetBytes(reader.Value));
                    writer?.WriteCData(reader.Value);
                    break;

                case XmlNodeType.Comment:
                    writer?.WriteComment(reader.Value);
                    break;

                case XmlNodeType.ProcessingInstruction:
                    writer?.WriteProcessingInstruction(reader.Name, reader.Value);
                    break;

                case XmlNodeType.XmlDeclaration:
                    if (writer is not null)
                    {
                        var standalone = reader.GetAttribute("standalone");
                        if (standalone is null)
                        {
                            writer.WriteStartDocument();
                        }
                        else
                        {
                            writer.WriteStartDocument(standalone == "yes");
                        }
                    }

                    break;
            }
        }

        return new Result(images, elements, Convert.ToHexString(textHash.GetHashAndReset()));
    }

    /// <summary>Copies the current element's start tag, replacing one attribute value or remapping relationship ids.</summary>
    private static void CopyElement(XmlReader reader, XmlWriter writer, Dictionary<string, string>? remapIds,
        string? attrNs, string? attrLocal, string? newValue, bool empty)
    {
        writer.WriteStartElement(reader.Prefix, reader.LocalName, reader.NamespaceURI);
        if (reader.MoveToFirstAttribute())
        {
            do
            {
                var value = reader.Value;
                if (newValue is not null && reader.LocalName == attrLocal && reader.NamespaceURI == attrNs)
                {
                    value = newValue;
                }
                else if (remapIds is not null && (reader.NamespaceURI == R || reader.NamespaceURI == O)
                         && remapIds.TryGetValue(value, out var mapped))
                {
                    value = mapped;
                }

                writer.WriteAttributeString(reader.Prefix, reader.LocalName, reader.NamespaceURI, value);
            }
            while (reader.MoveToNextAttribute());

            reader.MoveToElement();
        }

        if (empty)
        {
            writer.WriteEndElement();
        }
    }

    private static long? ParseLong(string? value) =>
        long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result) ? result : null;

    /// <summary>Converts a VML style's width/height to EMU.</summary>
    private static (long?, long?) ParseVmlSize(string? style)
    {
        if (style is null)
        {
            return (null, null);
        }

        long? width = null, height = null;
        foreach (Match m in VmlSize.Matches(style))
        {
            if (!double.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number))
            {
                continue;
            }

            var emuPerUnit = m.Groups[3].Value.ToLowerInvariant() switch
            {
                "in" => 914400.0,
                "cm" => 360000.0,
                "mm" => 36000.0,
                "px" => 9525.0,
                _ => 12700.0, // pt (VML default unit)
            };
            var emu = (long)Math.Round(number * emuPerUnit);
            if (m.Groups[1].Value.Equals("width", StringComparison.OrdinalIgnoreCase))
            {
                width = emu;
            }
            else
            {
                height = emu;
            }
        }

        return (width, height);
    }
}
