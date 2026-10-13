using System.Text;
using DocumentFormat.OpenXml.Packaging;

namespace ImageQualityEnhancer.Tests;

/// <summary>Builds a .docx covering the image layouts the DOCX path must handle.</summary>
internal static class DocxSamples
{
    public const string BodyUri = "/word/document.xml";

    // Ordinals of the body's image references, in document order (fallback markup is not counted).
    public const int PhotoInTable = 0;
    public const int PhotoAnchored = 1;
    public const int CroppedJpeg = 2;
    public const int RotatedScan = 3;
    public const int AlternateContent = 4; // shows the shared photo; its mc:Fallback repeats the reference
    public const int SvgFallback = 5;
    public const int OlePreview = 6;
    public const int Linked = 7;
    public const int Emf = 8;
    public const int Gif = 9;

    public sealed record Built(string PhotoUri, string JpegUri, string ScanUri, string GifUri, string HeaderUri);

    private const string Namespaces =
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" " +
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" " +
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" " +
        "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" " +
        "xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\" " +
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" " +
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" xmlns:o=\"urn:schemas-microsoft-com:office:office\" " +
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" " +
        "xmlns:asvg=\"http://schemas.microsoft.com/office/drawing/2016/SVG/main\" mc:Ignorable=\"wp14\"";

    public static Built Build(string path, string imagesFolder)
    {
        using var doc = WordprocessingDocument.Create(path, DocumentFormat.OpenXml.WordprocessingDocumentType.Document);
        var main = doc.AddMainDocumentPart();

        var photo = AddImage(main, ImagePartType.Png, Path.Combine(imagesFolder, "photo.png"));
        var jpeg = AddImage(main, ImagePartType.Jpeg, Path.Combine(imagesFolder, "photo.jpg"));
        var scan = AddImage(main, ImagePartType.Png, Path.Combine(imagesFolder, "scan.png"));
        var gif = AddImage(main, ImagePartType.Gif, Path.Combine(imagesFolder, "small.gif"));
        var svgPng = AddImage(main, ImagePartType.Png, Path.Combine(imagesFolder, "scan.png"));
        var svg = AddBytes(main, ImagePartType.Svg, "<svg xmlns=\"http://www.w3.org/2000/svg\"/>"u8.ToArray());
        var olePreview = AddImage(main, ImagePartType.Png, Path.Combine(imagesFolder, "scan.png"));
        var emf = AddBytes(main, ImagePartType.Emf, new byte[] { 1, 0, 0, 0, 108, 0, 0, 0 }.Concat(new byte[100]).ToArray());
        var linked = main.AddExternalRelationship(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", new Uri("file:///C:/images/outside.png")).Id;

        var header = main.AddNewPart<HeaderPart>();
        var headerPhoto = header.CreateRelationshipToPart(main.GetPartById(photo));
        WriteXml(header, $"<w:hdr {Namespaces}><w:p><w:r><w:t>Company header</w:t></w:r></w:p>{Inline(headerPhoto, 30, 600000, 400000)}</w:hdr>");

        var body = new StringBuilder();
        body.Append("<w:p><w:r><w:t xml:space=\"preserve\">Pharmaceutical report </w:t></w:r><w:r><w:t>with figures.</w:t></w:r></w:p>");
        body.Append("<w:tbl><w:tblPr><w:tblW w:w=\"5000\" w:type=\"pct\"/></w:tblPr><w:tblGrid><w:gridCol w:w=\"4000\"/><w:gridCol w:w=\"4000\"/></w:tblGrid>");
        body.Append("<w:tr><w:tc><w:p><w:r><w:t>Compound A</w:t></w:r></w:p></w:tc><w:tc>");
        body.Append(Inline(photo, 1, 1828800, 1371600));
        body.Append("</w:tc></w:tr></w:tbl>");
        body.Append(Anchor(photo, 2, 914400, 685800));
        body.Append(Inline(jpeg, 3, 1500000, 1000000, crop: "<a:srcRect l=\"10000\" t=\"5000\" r=\"10000\"/>"));
        body.Append(Inline(scan, 4, 2000000, 1500000, rotation: 5400000));
        body.Append("<w:p><w:r><mc:AlternateContent><mc:Choice Requires=\"wp14\"><w:drawing>")
            .Append(InlineDrawing(photo, 5, 900000, 600000))
            .Append("</w:drawing></mc:Choice><mc:Fallback><w:pict><v:shape style=\"width:70.85pt;height:47.25pt\">")
            .Append($"<v:imagedata r:id=\"{photo}\" o:title=\"\"/></v:shape></w:pict></mc:Fallback></mc:AlternateContent></w:r></w:p>");
        body.Append(Inline(svgPng, 6, 800000, 800000,
            blipChildren: $"<a:extLst><a:ext uri=\"{{96DAC541-7B7A-43D3-8B79-37D633B846F1}}\"><asvg:svgBlip r:embed=\"{svg}\"/></a:ext></a:extLst>"));
        body.Append("<w:p><w:r><w:object><v:shape style=\"width:120pt;height:80pt\">")
            .Append($"<v:imagedata r:id=\"{olePreview}\" o:title=\"\"/></v:shape>")
            .Append("<o:OLEObject Type=\"Embed\" ProgID=\"ChemDraw.Document.6.0\" ShapeID=\"_x0000_i1025\" DrawAspect=\"Content\" ObjectID=\"_1\"/></w:object></w:r></w:p>");
        body.Append(Inline(linked, 7, 500000, 500000, link: true));
        body.Append(Inline(emf, 8, 700000, 700000));
        body.Append(Inline(gif, 9, 600000, 450000));
        body.Append("<w:p><w:r><w:t>End of report.</w:t></w:r></w:p>");
        body.Append($"<w:sectPr><w:headerReference w:type=\"default\" r:id=\"{main.GetIdOfPart(header)}\"/></w:sectPr>");

        WriteXml(main, $"<w:document {Namespaces}><w:body>{body}</w:body></w:document>");

        return new Built(
            main.GetPartById(photo).Uri.OriginalString,
            main.GetPartById(jpeg).Uri.OriginalString,
            main.GetPartById(scan).Uri.OriginalString,
            main.GetPartById(gif).Uri.OriginalString,
            header.Uri.OriginalString);
    }

    private static string AddImage(MainDocumentPart main, PartTypeInfo type, string file) =>
        AddBytes(main, type, File.ReadAllBytes(file));

    private static string AddBytes(MainDocumentPart main, PartTypeInfo type, byte[] data)
    {
        var part = main.AddImagePart(type);
        using (var stream = new MemoryStream(data))
        {
            part.FeedData(stream);
        }

        return main.GetIdOfPart(part);
    }

    private static void WriteXml(OpenXmlPart part, string xml)
    {
        using var stream = part.GetStream(FileMode.Create);
        var bytes = Encoding.UTF8.GetBytes("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>" + xml);
        stream.Write(bytes);
    }

    private static string Inline(string relId, int id, long cx, long cy, string crop = "", int rotation = 0,
        string blipChildren = "", bool link = false) =>
        $"<w:p><w:r><w:drawing>{InlineDrawing(relId, id, cx, cy, crop, rotation, blipChildren, link)}</w:drawing></w:r></w:p>";

    private static string InlineDrawing(string relId, int id, long cx, long cy, string crop = "", int rotation = 0,
        string blipChildren = "", bool link = false) =>
        $"<wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\"><wp:extent cx=\"{cx}\" cy=\"{cy}\"/>" +
        $"<wp:docPr id=\"{id}\" name=\"Picture {id}\" descr=\"Figure {id}\"/>{Graphic(relId, id, cx, cy, crop, rotation, blipChildren, link)}</wp:inline>";

    private static string Anchor(string relId, int id, long cx, long cy) =>
        "<w:p><w:r><w:drawing><wp:anchor distT=\"0\" distB=\"0\" distL=\"114300\" distR=\"114300\" simplePos=\"0\" relativeHeight=\"2\" " +
        "behindDoc=\"0\" locked=\"0\" layoutInCell=\"1\" allowOverlap=\"1\"><wp:simplePos x=\"0\" y=\"0\"/>" +
        "<wp:positionH relativeFrom=\"column\"><wp:posOffset>1000000</wp:posOffset></wp:positionH>" +
        "<wp:positionV relativeFrom=\"paragraph\"><wp:posOffset>200000</wp:posOffset></wp:positionV>" +
        $"<wp:extent cx=\"{cx}\" cy=\"{cy}\"/><wp:wrapSquare wrapText=\"bothSides\"/>" +
        $"<wp:docPr id=\"{id}\" name=\"Floating {id}\"/>{Graphic(relId, id, cx, cy, "", 0, "", false)}</wp:anchor></w:drawing></w:r></w:p>";

    private static string Graphic(string relId, int id, long cx, long cy, string crop, int rotation, string blipChildren, bool link)
    {
        var blipAttr = link ? $"r:link=\"{relId}\"" : $"r:embed=\"{relId}\"";
        var blip = blipChildren.Length > 0 ? $"<a:blip {blipAttr}>{blipChildren}</a:blip>" : $"<a:blip {blipAttr}/>";
        var rot = rotation != 0 ? $" rot=\"{rotation}\"" : "";
        return "<a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\"><pic:pic>" +
               $"<pic:nvPicPr><pic:cNvPr id=\"{id}\" name=\"p{id}\"/><pic:cNvPicPr/></pic:nvPicPr>" +
               $"<pic:blipFill>{blip}{crop}<a:stretch><a:fillRect/></a:stretch></pic:blipFill>" +
               $"<pic:spPr><a:xfrm{rot}><a:off x=\"0\" y=\"0\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>" +
               "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>";
    }
}
