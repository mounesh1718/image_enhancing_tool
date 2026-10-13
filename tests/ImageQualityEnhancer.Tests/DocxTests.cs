using System.IO.Compression;
using System.Security.Cryptography;
using System.Xml.Linq;
using DocumentFormat.OpenXml.Packaging;
using ImageQualityEnhancer.Services;
using ImageQualityEnhancer.Services.Docx;

namespace ImageQualityEnhancer.Tests;

/// <summary>Sample images (generated once) and a running worker for real enhancement.</summary>
public sealed class DocxFixture : IAsyncLifetime
{
    private readonly TempDirectory _dir = new();

    public string ImagesFolder => Path.Combine(_dir.Path, "images");
    public PythonWorkerClient Worker { get; } = new();

    public async ValueTask InitializeAsync()
    {
        Directory.CreateDirectory(ImagesFolder);
        TestSamples.ImageSet(ImagesFolder);
        await Worker.StartAsync();
    }

    public async ValueTask DisposeAsync()
    {
        await Worker.DisposeAsync();
        _dir.Dispose();
    }
}

public class DocxTests : IClassFixture<DocxFixture>, IDisposable
{
    private const string Body = DocxSamples.BodyUri;
    private readonly DocxFixture _fixture;
    private readonly TempDirectory _root = new();
    private readonly JobWorkspace _ws;
    private readonly string _source;
    private readonly DocxSamples.Built _built;

    public DocxTests(DocxFixture fixture)
    {
        _fixture = fixture;
        _ws = new WorkspaceManager(_root.Path).Create();
        _source = _ws.Resolve(WorkspaceFolder.Original, "report.docx");
        _built = DocxSamples.Build(_source, fixture.ImagesFolder);
    }

    public void Dispose() => _root.Dispose();

    private static DocxImageOccurrence At(DocxScanResult scan, string part, int ordinal) =>
        scan.Occurrences.Single(o => o.PartUri == part && o.Ref.Ordinal == ordinal);

    // ---------------------------------------------------------------- scanning

    [Fact]
    public void Scan_FindsEveryImageWithItsLayout()
    {
        var scan = DocxScanner.Scan(_source);

        Assert.Equal(10, scan.Occurrences.Count(o => o.PartUri == Body)); // fallback copy not counted
        Assert.Single(scan.Occurrences, o => o.PartUri == _built.HeaderUri);

        var inTable = At(scan, Body, DocxSamples.PhotoInTable);
        Assert.True(inTable.Ref.InTable);
        Assert.Equal((1828800L, 1371600L), (inTable.Ref.ExtentCx!.Value, inTable.Ref.ExtentCy!.Value));
        Assert.Equal("1", inTable.Ref.DrawingId);

        Assert.True(At(scan, Body, DocxSamples.PhotoAnchored).Ref.Anchored);
        Assert.Equal([10000, 5000, 10000, 0], At(scan, Body, DocxSamples.CroppedJpeg).Ref.Crop!);
        Assert.Equal(_built.PhotoUri, At(scan, Body, DocxSamples.AlternateContent).ImageUri);
        Assert.Null(At(scan, Body, DocxSamples.Gif).SkipReason);
    }

    [Fact]
    public void Scan_MarksImagesThatMustStayUnchanged()
    {
        var scan = DocxScanner.Scan(_source);
        Assert.Equal("svg_fallback", At(scan, Body, DocxSamples.SvgFallback).SkipReason);
        Assert.Equal("ole_preview", At(scan, Body, DocxSamples.OlePreview).SkipReason);
        Assert.Equal("linked_image", At(scan, Body, DocxSamples.Linked).SkipReason);
        Assert.Equal("vector", At(scan, Body, DocxSamples.Emf).SkipReason);
        Assert.Equal(DocxImageKind.OlePreview, At(scan, Body, DocxSamples.OlePreview).Ref.Kind);
    }

    [Fact]
    public void Scan_CountsSharedImageReferencesAcrossParts()
    {
        var scan = DocxScanner.Scan(_source);
        var photo = scan.Images.Single(i => i.Uri == _built.PhotoUri);
        Assert.Equal(4, photo.ReferenceCount); // table, floating, compatibility block, header
        Assert.Equal("png", photo.OutputFormat);
        Assert.Equal("png", scan.Images.Single(i => i.Uri == _built.GifUri).OutputFormat);
    }

    [Fact]
    public void ExtractImage_CopiesStoredBytesExactly()
    {
        var target = _ws.Resolve(WorkspaceFolder.Extracted, "photo.png");
        DocxScanner.ExtractImage(_source, _built.PhotoUri, target);
        Assert.Equal(File.ReadAllBytes(Path.Combine(_fixture.ImagesFolder, "photo.png")), File.ReadAllBytes(target));
    }

    // ---------------------------------------------------------------- export

    private async Task<string> Enhance(string imageUri, string format)
    {
        var name = Path.GetFileNameWithoutExtension(imageUri);
        var extracted = _ws.Resolve(WorkspaceFolder.Extracted, name + Path.GetExtension(imageUri));
        if (!File.Exists(extracted))
        {
            DocxScanner.ExtractImage(_source, imageUri, extracted);
        }

        var output = _ws.Resolve(WorkspaceFolder.Enhanced, $"{name}.{(format == "jpeg" ? "jpg" : format)}");
        var result = await _fixture.Worker.SendAsync("enhance", new
        {
            input = extracted,
            output,
            format,
            settings = new { mode = "manual", sharpenAmount = 0.8 },
        });
        Assert.Equal("completed", result.GetProperty("status").GetString());
        return output;
    }

    private string Output => _ws.Resolve(WorkspaceFolder.Export, "report.docx");

    private static byte[] Sha(string path) => SHA256.HashData(File.ReadAllBytes(path));

    private static byte[] PartBytes(string docx, string uri)
    {
        using var zip = ZipFile.OpenRead(docx);
        using var stream = zip.GetEntry(uri.TrimStart('/'))!.Open();
        using var copy = new MemoryStream();
        stream.CopyTo(copy);
        return copy.ToArray();
    }

    private static string BodyText(string docx)
    {
        using var doc = WordprocessingDocument.Open(docx, false);
        var main = doc.MainDocumentPart!;
        return main.Document!.Body!.InnerText + "|" + string.Join("|", main.HeaderParts.Select(h => h.Header!.InnerText));
    }

    [Fact]
    public async Task Export_AllOccurrencesApproved_ReplacesInPlace_AndKeepsEverythingElse()
    {
        var originalHash = Sha(_source);
        var scan = DocxScanner.Scan(_source);
        var photo = await Enhance(_built.PhotoUri, "png");
        var jpeg = await Enhance(_built.JpegUri, "jpeg");
        var replacements = scan.Occurrences
            .Where(o => o.ImageUri == _built.PhotoUri)
            .Select(o => new DocxReplacement(o.PartUri, o.Ref.Ordinal, _built.PhotoUri, photo))
            .Append(new DocxReplacement(Body, DocxSamples.CroppedJpeg, _built.JpegUri, jpeg))
            .ToList();

        var result = DocxExporter.Export(_source, Output, replacements);

        Assert.True(result.Validation.Ok, string.Join("; ", result.Validation.Problems));
        Assert.Empty(result.Failed);
        Assert.Equal(2, result.ImagesReplacedInPlace);
        Assert.Equal(0, result.ImagesAdded);
        Assert.Equal(originalHash, Sha(_source));
        Assert.Equal(File.ReadAllBytes(photo), PartBytes(Output, _built.PhotoUri));
        Assert.Equal(PartBytes(_source, _built.ScanUri), PartBytes(Output, _built.ScanUri));
        Assert.Equal(PartBytes(_source, Body), PartBytes(Output, Body)); // no XML rewritten for in-place swaps
        Assert.Equal(BodyText(_source), BodyText(Output));
        Assert.Equal(["report.docx"], Directory.GetFiles(_ws.GetFolder(WorkspaceFolder.Export)).Select(Path.GetFileName));
    }

    [Fact]
    public async Task Export_SharedImagePartlyApproved_OnlyApprovedOccurrenceChanges()
    {
        var photo = await Enhance(_built.PhotoUri, "png");
        var result = DocxExporter.Export(_source, Output,
            [new DocxReplacement(Body, DocxSamples.PhotoInTable, _built.PhotoUri, photo)]);

        Assert.True(result.Validation.Ok, string.Join("; ", result.Validation.Problems));
        Assert.Equal(1, result.ImagesAdded);
        var after = DocxScanner.Scan(Output);
        var changed = At(after, Body, DocxSamples.PhotoInTable);
        Assert.NotEqual(_built.PhotoUri, changed.ImageUri);
        Assert.Equal(File.ReadAllBytes(photo), PartBytes(Output, changed.ImageUri!));
        // The other three occurrences (floating, compatibility block, header) still show the untouched original.
        Assert.Equal(_built.PhotoUri, At(after, Body, DocxSamples.PhotoAnchored).ImageUri);
        Assert.Equal(_built.PhotoUri, At(after, Body, DocxSamples.AlternateContent).ImageUri);
        Assert.Equal(_built.PhotoUri, after.Occurrences.Single(o => o.PartUri == _built.HeaderUri).ImageUri);
        Assert.Equal(PartBytes(_source, _built.PhotoUri), PartBytes(Output, _built.PhotoUri));
        Assert.Equal(BodyText(_source), BodyText(Output));
    }

    [Fact]
    public async Task Export_RedirectInCompatibilityBlock_AlsoUpdatesFallbackCopy()
    {
        var photo = await Enhance(_built.PhotoUri, "png");
        DocxExporter.Export(_source, Output, [new DocxReplacement(Body, DocxSamples.AlternateContent, _built.PhotoUri, photo)]);

        var xml = XDocument.Parse(System.Text.Encoding.UTF8.GetString(PartBytes(Output, Body)));
        XNamespace mc = "http://schemas.openxmlformats.org/markup-compatibility/2006";
        XNamespace a = "http://schemas.openxmlformats.org/drawingml/2006/main";
        XNamespace v = "urn:schemas-microsoft-com:vml";
        XNamespace r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        var block = xml.Descendants(mc + "AlternateContent").Single();
        var choiceId = block.Descendants(a + "blip").Single().Attribute(r + "embed")!.Value;
        var fallbackId = block.Descendants(v + "imagedata").Single().Attribute(r + "id")!.Value;
        Assert.StartsWith("rIdIqe", choiceId);
        Assert.Equal(choiceId, fallbackId);
    }

    [Fact]
    public async Task Export_Gif_IsStoredAsNewPngPart()
    {
        var gif = await Enhance(_built.GifUri, "png");
        var result = DocxExporter.Export(_source, Output, [new DocxReplacement(Body, DocxSamples.Gif, _built.GifUri, gif)]);

        Assert.True(result.Validation.Ok, string.Join("; ", result.Validation.Problems));
        var after = DocxScanner.Scan(Output);
        var image = after.Images.Single(i => i.Uri == At(after, Body, DocxSamples.Gif).ImageUri);
        Assert.Equal("image/png", image.ContentType);
        using var doc = WordprocessingDocument.Open(Output, false); // package (content types, rels) is valid
        Assert.NotNull(doc.MainDocumentPart!.Document!.Body);
    }

    [Fact]
    public async Task Export_InvalidRequestsFailAlone()
    {
        var photo = await Enhance(_built.PhotoUri, "png");
        var scanPng = await Enhance(_built.ScanUri, "png");
        var result = DocxExporter.Export(_source, Output,
        [
            new DocxReplacement(Body, 99, _built.PhotoUri, photo),                       // no such position
            new DocxReplacement(Body, DocxSamples.PhotoInTable, _built.ScanUri, photo),   // different image there
            new DocxReplacement(Body, DocxSamples.OlePreview, "/word/media/x.png", photo), // never replaceable
            new DocxReplacement(Body, DocxSamples.RotatedScan, _built.ScanUri, scanPng),  // valid
        ]);

        Assert.Equal(["missing_occurrence", "changed_occurrence", "changed_occurrence"], result.Failed.Select(f => f.Code));
        Assert.Single(result.Applied);
        Assert.True(result.Validation.Ok, string.Join("; ", result.Validation.Problems));
    }

    [Fact]
    public async Task Export_AddsNoSchemaErrors()
    {
        var photo = await Enhance(_built.PhotoUri, "png");
        var gif = await Enhance(_built.GifUri, "png");
        DocxExporter.Export(_source, Output,
        [
            new DocxReplacement(Body, DocxSamples.AlternateContent, _built.PhotoUri, photo),
            new DocxReplacement(Body, DocxSamples.Gif, _built.GifUri, gif),
        ]);

        static string[] Errors(string path)
        {
            using var doc = WordprocessingDocument.Open(path, false);
            return new DocumentFormat.OpenXml.Validation.OpenXmlValidator()
                .Validate(doc).Select(e => $"{e.Part?.Uri} {e.Path?.XPath} {e.Description}").Order().ToArray();
        }

        Assert.Equal(Errors(_source), Errors(Output));
    }

    [Fact]
    public void Export_ToSourcePath_IsRejected()
    {
        var ex = Assert.Throws<DocxException>(() => DocxExporter.Export(_source, _source, []));
        Assert.Equal("bad_request", ex.Code);
    }

    // ---------------------------------------------------------------- validation

    [Fact]
    public async Task Validator_DetectsUnapprovedImageChange()
    {
        var photo = await Enhance(_built.PhotoUri, "png");
        var scan = DocxScanner.Scan(_source);
        var all = scan.Occurrences.Where(o => o.ImageUri == _built.PhotoUri)
            .Select(o => new DocxReplacement(o.PartUri, o.Ref.Ordinal, _built.PhotoUri, photo)).ToList();
        DocxExporter.Export(_source, Output, all);

        // Pretend only one occurrence was approved: the other changed images must be reported.
        var result = DocxValidator.Validate(_source, Output, scan,
            new Dictionary<(string, int), string> { [(Body, DocxSamples.PhotoInTable)] = photo },
            new HashSet<string> { _built.PhotoUri.TrimStart('/') });
        Assert.False(result.Ok);
        Assert.Contains(result.Problems, p => p.Contains("was not approved"));
    }

    [Fact]
    public void Validator_DetectsTextChange()
    {
        var tampered = _ws.Resolve(WorkspaceFolder.Export, "tampered.docx");
        File.Copy(_source, tampered);
        using (var zip = ZipFile.Open(tampered, ZipArchiveMode.Update))
        {
            var entry = zip.GetEntry("word/document.xml")!;
            string xml;
            using (var reader = new StreamReader(entry.Open()))
            {
                xml = reader.ReadToEnd();
            }

            entry.Delete();
            using var writer = new StreamWriter(zip.CreateEntry("word/document.xml").Open());
            writer.Write(xml.Replace("Compound A", "Compound B"));
        }

        var result = DocxValidator.Validate(_source, tampered, DocxScanner.Scan(_source),
            new Dictionary<(string, int), string>(), new HashSet<string> { "word/document.xml" });
        Assert.False(result.Ok);
        Assert.Contains(result.Problems, p => p.Contains("text changed"));
    }

    // ---------------------------------------------------------------- unreadable files

    [Theory]
    [InlineData("missing", "file_not_found")]
    [InlineData("legacy", "legacy_doc")]
    [InlineData("encrypted", "password_protected")]
    [InlineData("garbage", "corrupt_docx")]
    [InlineData("zip", "corrupt_docx")]
    public void Scan_ExplainsUnreadableFiles(string kind, string code)
    {
        var path = _ws.Resolve(WorkspaceFolder.Original, kind + ".docx");
        byte[] cfb = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
        switch (kind)
        {
            case "legacy":
                File.WriteAllBytes(path, [.. cfb, .. new byte[4000]]);
                break;
            case "encrypted":
                File.WriteAllBytes(path, [.. cfb, .. new byte[500], .. System.Text.Encoding.Unicode.GetBytes("EncryptedPackage"), .. new byte[500]]);
                break;
            case "garbage":
                File.WriteAllBytes(path, "not a document"u8.ToArray());
                break;
            case "zip":
                using (var zip = ZipFile.Open(path, ZipArchiveMode.Create))
                {
                    using var w = new StreamWriter(zip.CreateEntry("hello.txt").Open());
                    w.Write("hi");
                }

                break;
        }

        var ex = Assert.Throws<DocxException>(() => DocxScanner.Scan(path));
        Assert.Equal(code, ex.Code);
    }
}
