using System.Security.Cryptography;
using System.Text.Json;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>End-to-end PDF path through the worker process: scan, extract, enhance, replace, validate.</summary>
public class WorkerPdfTests : IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(120);
    private readonly TempDirectory _root = new();
    private readonly JobWorkspace _workspace;

    public WorkerPdfTests()
    {
        _workspace = new WorkspaceManager(_root.Path).Create();
    }

    public void Dispose() => _root.Dispose();

    [Fact]
    public async Task FullPdfPipeline_ReplacesEligibleImages_AndKeepsOriginal()
    {
        var source = _workspace.Resolve(WorkspaceFolder.Original, "report.pdf");
        TestSamples.FeaturePdf(source);
        var originalHash = SHA256.HashData(await File.ReadAllBytesAsync(source));
        var progressEvents = 0;
        var progress = new Progress<JsonElement>(_ => Interlocked.Increment(ref progressEvents));

        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var scan = await client.SendAsync("pdf_scan", new { path = source }, progress).WaitAsync(Timeout);
        Assert.Equal(5, scan.GetProperty("pageCount").GetInt32());
        var eligible = scan.GetProperty("images").EnumerateArray()
            .Where(i => i.GetProperty("eligible").GetBoolean())
            .ToList();
        Assert.NotEmpty(eligible);

        var extracted = await client.SendAsync("pdf_extract", new
        {
            path = source,
            xrefs = eligible.Select(i => i.GetProperty("xref").GetInt32()),
            outputDir = _workspace.GetFolder(WorkspaceFolder.Extracted),
        }).WaitAsync(Timeout);

        var replacements = new List<object>();
        foreach (var item in extracted.GetProperty("images").EnumerateArray())
        {
            var xref = item.GetProperty("xref").GetInt32();
            var format = eligible.Single(i => i.GetProperty("xref").GetInt32() == xref).GetProperty("outputFormat").GetString()!;
            var output = _workspace.Resolve(WorkspaceFolder.Enhanced, $"x{xref}{(format == "jpeg" ? ".jpg" : ".png")}");
            var result = await client.SendAsync("enhance", new
            {
                input = item.GetProperty("path").GetString(),
                output,
                format,
                settings = new { mode = "manual", sharpenAmount = 0.5 },
            }).WaitAsync(Timeout);
            Assert.Equal("completed", result.GetProperty("status").GetString());
            replacements.Add(new { xref, path = output });
        }

        var exported = _workspace.Resolve(WorkspaceFolder.Export, "report.pdf");
        var replaced = await client.SendAsync("pdf_replace", new { source, output = exported, replacements }, progress)
            .WaitAsync(Timeout);

        Assert.True(replaced.GetProperty("validation").GetProperty("ok").GetBoolean());
        Assert.Equal(replacements.Count, replaced.GetProperty("replaced").GetArrayLength());
        Assert.True(File.Exists(exported));
        Assert.Equal(originalHash, SHA256.HashData(await File.ReadAllBytesAsync(source)));
        Assert.True(progressEvents > 0);
        Assert.Equal(["report.pdf"], Directory.GetFiles(_workspace.GetFolder(WorkspaceFolder.Export)).Select(Path.GetFileName));
    }

    [Fact]
    public async Task PasswordProtectedPdf_ReportsPasswordRequired()
    {
        var source = _workspace.Resolve(WorkspaceFolder.Original, "secret.pdf");
        TestSamples.FeaturePdf(source, "s3cret");
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var ex = await Assert.ThrowsAsync<WorkerException>(() => client.SendAsync("pdf_scan", new { path = source }));
        Assert.Equal("password_required", ex.Code);

        var scan = await client.SendAsync("pdf_scan", new { path = source, password = "s3cret" }).WaitAsync(Timeout);
        Assert.Equal(5, scan.GetProperty("pageCount").GetInt32());
    }

    [Fact]
    public async Task CorruptPdf_ReportsCorruptAndWorkerSurvives()
    {
        var source = _workspace.Resolve(WorkspaceFolder.Original, "broken.pdf");
        await File.WriteAllBytesAsync(source, "%PDF-1.7\n garbage garbage"u8.ToArray());
        await using var client = new PythonWorkerClient();
        await client.StartAsync();

        var ex = await Assert.ThrowsAsync<WorkerException>(() => client.SendAsync("pdf_scan", new { path = source }));
        Assert.Equal("corrupt_pdf", ex.Code);
        await client.SendAsync("ping").WaitAsync(Timeout);

        // The worker must not keep the user's file locked after a failed open.
        File.Delete(source);
    }
}
