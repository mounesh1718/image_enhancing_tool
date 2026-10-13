using System.Diagnostics;
using System.Runtime.CompilerServices;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.Tests;

/// <summary>
/// Generates test images and documents with the worker's Python test helpers (worker/tests/samples.py,
/// pdf_samples.py), so no binary fixtures are stored in the repository.
/// </summary>
internal static class TestSamples
{
    public static void NoisyPhoto(string path) =>
        RunPython("import samples; samples.write(sys.argv[1], samples.add_noise(samples.good_photo(), 12))", path);

    public static void FeaturePdf(string path, string? password = null) =>
        RunPython("import pdf_samples; pdf_samples.build_feature_pdf(sys.argv[1], sys.argv[2] or None)", path, password ?? "");

    /// <summary>Writes photo.png, photo.jpg, scan.png and small.gif (all noisy enough to need enhancement) into a folder.</summary>
    public static void ImageSet(string folder) =>
        RunPython(
            "import os, samples; d = sys.argv[1];" +
            "samples.write(os.path.join(d, 'photo.png'), samples.add_noise(samples.good_photo(), 12));" +
            "samples.write(os.path.join(d, 'photo.jpg'), samples.add_noise(samples.photo(200, 300, seed=3), 12));" +
            "samples.write(os.path.join(d, 'scan.png'), samples.faded_scan(samples.line_art()));" +
            "samples.write(os.path.join(d, 'small.gif'), samples.add_noise(samples.photo(120, 160, seed=9), 12))",
            folder);

    /// <summary>A 3-page TIFF of noisy photos.</summary>
    public static void MultiPageTiff(string path) =>
        RunPython(
            "import cv2, samples; pages = [samples.add_noise(samples.photo(120, 160, seed=s), 12) for s in (1, 2, 3)];" +
            "ok, buf = cv2.imencodemulti('.tif', pages); buf.tofile(sys.argv[1])",
            path);

    /// <summary>The feature PDF with a signature field added (as a signed document would have).</summary>
    public static void SignedPdf(string path) =>
        RunPython(
            "import pymupdf, pdf_samples; tmp = sys.argv[1] + '.src.pdf'; pdf_samples.build_feature_pdf(tmp);" +
            "doc = pymupdf.open(tmp); w = pymupdf.Widget(); w.field_type = pymupdf.PDF_WIDGET_TYPE_SIGNATURE;" +
            "w.field_name = 'Approval'; w.rect = pymupdf.Rect(400, 700, 550, 750); doc[0].add_widget(w);" +
            "doc.save(sys.argv[1]); doc.close(); import os; os.remove(tmp)",
            path);

    /// <summary>A chemical structure drawing made by RDKit (so the right molecule is known).</summary>
    public static void StructureDrawing(string path, string smiles) =>
        RunPython("import cv2, test_redraw; cv2.imwrite(sys.argv[1], test_redraw.drawing(sys.argv[2]))", path, smiles);

    public static void OrdinaryTable(string path) =>
        RunPython("import cv2, test_redraw; cv2.imwrite(sys.argv[1], test_redraw.ordinary_table())", path);

    /// <summary>True when the picture classifier (DocumentFigureClassifier ONNX model) is installed for the worker's Python.</summary>
    public static bool ClassifierInstalled()
    {
        var psi = new ProcessStartInfo(new WorkerOptions().PythonExecutable) { UseShellExecute = false, RedirectStandardError = true };
        psi.ArgumentList.Add("-B");
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add($"import sys; sys.path.insert(0, {PyString(SourceWorkerDir())}); "
                             + "import figure_classifier as f; sys.exit(0 if f.available() else 1)");
        using var process = Process.Start(psi)!;
        process.StandardError.ReadToEnd();
        process.WaitForExit();
        return process.ExitCode == 0;
    }

    /// <summary>One picture of each group in a folder: table.png, chart.png, photo.png, structure.png (RDKit drawing
    /// of D-penicillamine; left out when RDKit is not installed), and unsure.png (a faded scan of line art the
    /// classifier is unsure about). Returns the files written.</summary>
    public static List<string> CategorySet(string folder)
    {
        Directory.CreateDirectory(folder);
        RunPython(
            "import os, cv2, samples, test_redraw; from test_classify import line_chart; d = sys.argv[1];" +
            "cv2.imwrite(os.path.join(d, 'table.png'), test_redraw.ordinary_table());" +
            "cv2.imwrite(os.path.join(d, 'chart.png'), line_chart());" +
            "cv2.imwrite(os.path.join(d, 'photo.png'), samples.photo());" +
            "cv2.imwrite(os.path.join(d, 'unsure.png'), samples.faded_scan(samples.line_art()));\n" +
            "try:\n    cv2.imwrite(os.path.join(d, 'structure.png'), test_redraw.drawing('CC(C)(S)[C@@H](N)C(=O)O'))\n" +
            "except Exception:\n    pass",
            folder);
        return Directory.GetFiles(folder, "*.png").OrderBy(f => f, StringComparer.Ordinal).ToList();
    }

    /// <summary>True when the optional redraw (MolScribe model, PyTorch, RDKit) is installed for the worker's Python.</summary>
    public static bool RedrawInstalled()
    {
        var psi = new ProcessStartInfo(new WorkerOptions().PythonExecutable) { UseShellExecute = false, RedirectStandardError = true };
        psi.ArgumentList.Add("-B");
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add($"import sys; sys.path.insert(0, {PyString(SourceWorkerDir())}); "
                             + "from redraw import molscribe_reader as m; import rdkit, torch; sys.exit(0 if m.model_path().exists() else 1)");
        using var process = Process.Start(psi)!;
        process.StandardError.ReadToEnd();
        process.WaitForExit();
        return process.ExitCode == 0;
    }

    /// <summary>A structure drawing (aspirin) with a caption under it: never redrawn (the caption would be lost).</summary>
    public static void StructureWithCaption(string path) =>
        RunPython("import cv2, test_redraw; img = test_redraw.drawing('CC(=O)Oc1ccccc1C(=O)O');"
                  + "img = cv2.copyMakeBorder(img, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255));"
                  + "cv2.putText(img, 'Figure 3: Aspirin tablets', (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA);"
                  + "cv2.imwrite(sys.argv[1], img)", path);

    /// <summary>True when the AI repair models (SCUNet, FBCNN, DocRes) and OCR are installed for the worker's Python.</summary>
    public static bool AiRepairInstalled()
    {
        var psi = new ProcessStartInfo(new WorkerOptions().PythonExecutable) { UseShellExecute = false, RedirectStandardError = true };
        psi.ArgumentList.Add("-B");
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add($"import sys; sys.path.insert(0, {PyString(SourceWorkerDir())}); "
                             + "import restore_ai, textcheck; sys.exit(0 if all(restore_ai.installed(k) for k in restore_ai.MODELS) "
                             + "and textcheck.available() else 1)");
        using var process = Process.Start(psi)!;
        process.StandardError.ReadToEnd();
        process.WaitForExit();
        return process.ExitCode == 0;
    }

    /// <summary>A grey text label with heavy grain (as a poor scan or phone photo of a page).</summary>
    public static void GrainyText(string path) =>
        RunPython("import cv2, numpy as np; img = np.full((230, 640, 3), 235, np.uint8);"
                  + "[cv2.putText(img, t, (20, 50 + i * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA) "
                  + "for i, t in enumerate(['Batch 4521: assay 99.3 % w/w', 'Water content 0.42 % (KF)', 'Lot 7781  Exp 03/2028'])];"
                  + "img = np.clip(img + np.random.default_rng(2).normal(0, 30, img.shape), 0, 255).astype(np.uint8);"
                  + "cv2.imwrite(sys.argv[1], img)", path);

    private static void RunPython(string code, params string[] args)
    {
        var worker = SourceWorkerDir();
        var psi = new ProcessStartInfo(new WorkerOptions().PythonExecutable)
        {
            UseShellExecute = false,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add("-B");
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add($"import sys; sys.path[:0] = [{PyString(worker)}, {PyString(Path.Combine(worker, "tests"))}]; {code}");
        foreach (var arg in args)
        {
            psi.ArgumentList.Add(arg);
        }

        using var process = Process.Start(psi)!;
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, stderr);
    }

    private static string PyString(string value) => "r'" + value.Replace("'", "") + "'";

    /// <summary>The test helpers are not shipped with the app, so use the source tree's worker folder.</summary>
    private static string SourceWorkerDir([CallerFilePath] string thisFile = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(thisFile)!, "..", "..", "worker"));
}
