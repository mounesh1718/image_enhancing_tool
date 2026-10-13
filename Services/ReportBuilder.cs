using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using ImageQualityEnhancer.Models;

namespace ImageQualityEnhancer.Services;

/// <summary>How faithful an enhanced image is to its original, as shown in the report.</summary>
public enum AccuracyRating
{
    /// <summary>A redrawn structure that passed every automatic check (see the redraw evidence). Not proof that it
    /// matches the original: a chemist compares and approves it.</summary>
    Verified,
    High,
    Good,
    Check,
}

/// <summary>
/// Builds a self-contained HTML report of a job: for every image, whether it was re-altered or kept unchanged (and
/// why), the steps applied, quality before and after, and how faithful the result is to the original.
/// The tool never redraws images, so the report states that no image was redrawn.
/// </summary>
public static class ReportBuilder
{
    /// <summary>Thumbnails are embedded for at most this many images to keep the file a reasonable size.</summary>
    public const int MaxEmbeddedThumbnails = 400;

    /// <summary>
    /// Content accuracy is judged mainly by how many of the original's edges (lines, letters, shapes) are still there and
    /// whether new edges appeared; structural similarity is shown too but is naturally lower when a background is cleaned.
    /// </summary>
    public static AccuracyRating Rate(ImageJob job)
    {
        if (job.IsRedrawn)
        {
            // Its accuracy is the chemical proof, not a pixel comparison; a redraw without full proof must be checked.
            return job.IsUnprovenRedraw ? AccuracyRating.Check : AccuracyRating.Verified;
        }

        if (job.UseAiVersion)
        {
            // The AI version passed its own checks; it is rated on how much of the original it keeps.
            return job.Ai is { Fidelity: { } af } && af.EdgesKept >= 0.95 && af.NewEdges <= 0.15 ? AccuracyRating.Good : AccuracyRating.Check;
        }

        if (job.Fidelity is not { } f)
        {
            return AccuracyRating.Check;
        }

        if (job.TextCheck is { Flag: true })
        {
            return AccuracyRating.Check; // text reads differently after enhancement
        }

        var flagged = job.ReviewReasons.Any(r => r is "possible_artifacts" or "possible_detail_loss" or "heavy_blur" or "large_change" or "ai_restored");
        if (f.EdgesKept >= 0.95 && f.NewEdges <= 0.15 && !flagged)
        {
            return AccuracyRating.High;
        }

        return f.EdgesKept >= 0.85 && f.NewEdges <= 0.30 ? AccuracyRating.Good : AccuracyRating.Check;
    }

    public static string Build(JobState state, string workspaceRoot, DateTimeOffset? now = null)
    {
        var images = state.Images;
        var realtered = images.Where(i => i.Status == ProcessingStatus.Completed && !i.IsRedrawn).ToList();
        var redrawn = images.Where(i => i.IsRedrawn).ToList();
        var unchanged = images.Where(i => i.Status == ProcessingStatus.Skipped).ToList();
        var failed = images.Where(i => i.Status == ProcessingStatus.Failed).ToList();
        var approved = images.Count(i => i.Approval == ApprovalStatus.Approved);
        var ratings = realtered.GroupBy(Rate).ToDictionary(g => g.Key, g => g.Count());

        var html = new StringBuilder();
        html.Append("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
        html.Append("<title>Image enhancement report – ").Append(E(state.SourceFileName)).Append("</title><style>").Append(Css).Append("</style></head><body><main>");

        html.Append("<header><p class=\"eyebrow\">Image enhancement report</p><h1>").Append(E(state.SourceFileName)).Append("</h1>");
        html.Append("<p class=\"muted\">Created ").Append(E((now ?? DateTimeOffset.Now).ToLocalTime().ToString("d MMMM yyyy, HH:mm", CultureInfo.InvariantCulture)))
            .Append(" · Image Quality Enhancer · ")
            .Append(images.Any(i => i.Settings.Redraw && i.Settings.RedrawAll)
                ? "quality improvement, and redraw of every chemical structure the reader could read (redraws that are not verified are marked)"
                : images.Any(i => i.Settings.Redraw)
                    ? "quality improvement, and redraw of chemical structures where every automatic check passed"
                    : "quality improvement only (no image is redrawn)")
            .Append("</p></header>");

        // Summary
        html.Append("<section class=\"tiles\">");
        Tile(html, images.Count.ToString(CultureInfo.InvariantCulture), "images found");
        Tile(html, realtered.Count.ToString(CultureInfo.InvariantCulture), "re-altered (enhanced)");
        Tile(html, unchanged.Count.ToString(CultureInfo.InvariantCulture), "kept unchanged");
        Tile(html, failed.Count.ToString(CultureInfo.InvariantCulture), "failed (original kept)");
        Tile(html, redrawn.Count.ToString(CultureInfo.InvariantCulture), "redrawn (chemical structures)");
        Tile(html, approved.ToString(CultureInfo.InvariantCulture), "approved for the new copy");
        html.Append("</section>");

        // Optional AI enhancement, tried by the user after processing (each shared picture counted once)
        var aiTried = images.Where(i => i.Ai is not null).GroupBy(i => i.ResourceRef).Select(g => g.First()).ToList();
        if (aiTried.Count > 0)
        {
            html.Append("<p class=\"muted\">AI enhancement (optional, tried after processing): <b>").Append(aiTried.Count)
                .Append("</b> tried · <b>").Append(aiTried.Count(i => i.Ai!.Accepted)).Append("</b> usable · <b>")
                .Append(aiTried.Count(i => i.Ai!.Status == "rejected")).Append("</b> not usable · <b>")
                .Append(aiTried.Count(i => i.UseAiVersion)).Append("</b> chosen instead of the OpenCV result.</p>");
        }

        // Groups from the picture classifier (a sheet with several kinds of figures counts in each group)
        html.Append("<p class=\"muted\">Groups: <b>")
            .Append(images.Count(i => i.Groups.Contains(ImageCategory.Chemical))).Append("</b> chemical diagrams · <b>")
            .Append(images.Count(i => i.Groups.Contains(ImageCategory.FlowchartOrGraph))).Append("</b> flowcharts and graphs · <b>")
            .Append(images.Count(i => i.Groups.Contains(ImageCategory.Table))).Append("</b> tables · <b>")
            .Append(images.Count(i => i.Groups.Contains(ImageCategory.Other))).Append("</b> other images")
            .Append(images.Any(i => i.IsPossibleChemical) ? $" ({images.Count(i => i.IsPossibleChemical)} possible chemical diagrams need review)" : "")
            .Append(".</p>");

        if (realtered.Count > 0)
        {
            html.Append("<p class=\"muted\">Accuracy of the enhanced images: <b>")
                .Append(ratings.GetValueOrDefault(AccuracyRating.High)).Append(" high</b>, <b>")
                .Append(ratings.GetValueOrDefault(AccuracyRating.Good)).Append(" good</b>, <b>")
                .Append(ratings.GetValueOrDefault(AccuracyRating.Check)).Append(" to check</b>.</p>");
        }

        // How accuracy is checked
        html.Append("<section class=\"explain\"><h2>How accuracy is ensured</h2><ul>")
            .Append("<li><b>Re-alter.</b> Each step changes the brightness or colour of existing pixels; the pixel size never changes. Nothing is drawn or invented, and text, values and shapes are never replaced.</li>")
            .Append("<li><b>Groups.</b> DocumentFigureClassifier sorts every extracted picture (and each panel of a sheet of figures) into chemical diagrams, flowcharts and graphs, tables, and other images (photos included). Its suggestion and how sure it is are shown, and you can change the category. A picture classified as a chemical diagram gets a second check: MolScribe tries to read its atoms and bonds. <i>Classified as chemical</i> is not the same as <i>chemical structure verified</i>: a picture MolScribe cannot read reliably stays with the chemical diagrams, marked <i>possible chemical diagram, needs review</i>. MolScribe never reads tables, graphs, photos or other images.</li>")
            .Append("<li><b>Redraw (only when switched on, chemical diagrams only).</b> A structure is redrawn only when every automatic check passes: MolScribe reads it; every bond, wedge and label is found in the picture itself; OCR confirms the labels and finds no other text; a second reading at another scale gives the same molecule; and the clean redraw reads back as the same molecule (same InChIKey, including stereochemistry). Such a redraw is marked <span class=\"pill verified\">Verified</span>. These checks compare the redraw with what was read from the picture: they do not prove that it matches the original, so a chemist must compare and approve every redraw, and none is ever approved automatically.</li>")
            .Append("<li><b>Measured against the original.</b> Every result is compared with its original: <i>edges kept</i> (lines, letters and shapes still present), <i>new edges</i> (possible artefacts), <i>structural similarity</i> (SSIM, 100 % = identical structure) and brightness change.</li>")
            .Append("<li><b>Text check.</b> Images with text are read by OCR before and after. Only characters read the same way twice in the original are compared, and any changed number or more than 3 % changed characters marks the image <i>Check</i>.</li>")
            .Append("<li><b>Rating.</b> <span class=\"pill high\">High</span> 95 %+ of edges kept, almost no new edges, nothing flagged. <span class=\"pill good\">Good</span> 85 %+ kept. <span class=\"pill check\">Check</span> anything else, or flagged: look at it zoomed in before approving.</li>")
            .Append("<li><b>Your approval.</b> Only images you approve go into the new copy. The original file is never modified.</li>")
            .Append("</ul><p class=\"muted\">Structural similarity is naturally lower when a grey or speckled background is cleaned to white. That is a wanted change, so the rating relies mainly on edges kept.</p></section>");

        // Per image
        html.Append("<h2>Images</h2>");
        var embedded = 0;
        foreach (var job in images)
        {
            var embed = embedded++ < MaxEmbeddedThumbnails;
            AppendImage(html, job, workspaceRoot, embed);
        }

        if (images.Count > MaxEmbeddedThumbnails)
        {
            html.Append("<p class=\"muted\">Thumbnails are shown for the first ").Append(MaxEmbeddedThumbnails)
                .Append(" images to keep this file small; all images are listed.</p>");
        }

        html.Append("</main></body></html>");
        return html.ToString();
    }

    private static void AppendImage(StringBuilder html, ImageJob job, string root, bool embed)
    {
        var (badgeClass, badge) = job.Status switch
        {
            _ when job.UseAiVersion => ("check", "AI version"),
            ProcessingStatus.Completed when job.IsRedrawn => ("verified", "Redrawn"),
            ProcessingStatus.Completed => ("good", "Re-altered"),
            ProcessingStatus.Skipped => ("muted", "Kept unchanged"),
            ProcessingStatus.Failed => ("check", "Failed · original kept"),
            _ => ("muted", job.Status.ToString()),
        };

        html.Append("<article class=\"image\"><div class=\"thumbs\">");
        Thumb(html, embed ? job.OriginalThumbnailPath : null, root, "Original");
        Thumb(html, embed && job.Status == ProcessingStatus.Completed ? job.EnhancedThumbnailPath : null, root,
            job.IsRedrawn ? "Redrawn" : "Enhanced");
        html.Append("</div><div class=\"body\"><div class=\"title\"><h3>").Append(E(Wording.Location(job))).Append("</h3>")
            .Append("<span class=\"pill ").Append(badgeClass).Append("\">").Append(badge).Append("</span>");
        if (job.Status == ProcessingStatus.Completed)
        {
            var rating = Rate(job);
            html.Append("<span class=\"pill ").Append(rating.ToString().ToLowerInvariant()).Append("\">Accuracy: ").Append(rating).Append("</span>");
        }

        html.Append("</div><p class=\"muted\"><b>Category:</b> ").Append(E(Wording.CategoryLine(job)));
        if (job.IsPossibleChemical)
        {
            html.Append(" · <b>possible chemical diagram, needs review</b>");
        }

        html.Append("</p>");
        if (Wording.ChemicalStatus(job) is { Length: > 0 } chemistry)
        {
            html.Append("<p class=\"note\">").Append(E(chemistry).Replace("\n", "<br>")).Append("</p>");
        }

        html.Append("<p class=\"muted\">").Append(E(job.Id)).Append(" · ").Append(Wording.ContentKind(job.ContentKind));
        if (job.PixelSize is { } px)
        {
            html.Append(" · ").Append(px.Width).Append('×').Append(px.Height).Append(" px");
            if (job.OutputPixelSize is { } outPx && (outPx.Width != px.Width || outPx.Height != px.Height))
            {
                html.Append(" → ").Append(outPx.Width).Append('×').Append(outPx.Height).Append(" px");
            }
        }

        if (job.Status == ProcessingStatus.Completed)
        {
            html.Append(" · ").Append(job.Settings.Mode == "auto" ? Wording.Level(job.Settings.Strength) : "manual settings");
        }

        html.Append(" · Decision: ").Append(job.Approval switch
        {
            ApprovalStatus.Approved => "approved",
            ApprovalStatus.KeepOriginal => "original kept",
            _ => job.Status == ProcessingStatus.Completed ? "not decided yet" : "—",
        }).Append("</p>");

        if (job.Ai is { } ai)
        {
            var normal = job.Status == ProcessingStatus.Completed ? "the OpenCV result" : "the original (nothing needed fixing)";
            html.Append("<p class=\"note\"><b>Version chosen:</b> ")
                .Append(job.UseAiVersion ? $"the AI version ({E(ai.ModelNames)})" : normal).Append(". ")
                .Append(E(Wording.AiSummary(job))).Append("</p>");
        }

        if (job.Status != ProcessingStatus.Completed && !job.UseAiVersion)
        {
            html.Append("<p><b>Why:</b> ").Append(E(Capitalise(job.Error) ?? "—")).Append("</p>");
            if (job.Redraw is { Redrawn: false } notRedrawn && Wording.HasChemicalDetails(notRedrawn, job.ContentKind))
            {
                html.Append("<p class=\"muted\"><b>Not redrawn:</b> ").Append(E(string.Join("; ", notRedrawn.Reasons))).Append("</p>");
            }

            html.Append("</div></article>");
            return;
        }

        // Steps
        html.Append("<p><b>Process applied").Append(job.UseAiVersion ? " (AI version)" : "").Append(":</b></p><ol class=\"steps\">");
        foreach (var op in job.UseAiVersion && job.Ai is { } chosenAi ? chosenAi.Operations : job.Operations)
        {
            html.Append("<li><b>").Append(E(Capitalise(Wording.Operation(op)))).Append("</b> – ").Append(E(Wording.OperationExplained(op))).Append("</li>");
        }

        html.Append("</ol>");

        // Quality before / after
        if (job.QualityBefore is { } b && job.QualityAfter is { } a)
        {
            html.Append("<table><thead><tr><th>Quality</th><th>Before</th><th>After</th><th></th></tr></thead><tbody>");
            Row(html, "Sharpness (edge strength)", b.Sharpness, a.Sharpness, higherIsBetter: true, "0");
            Row(html, "Noise level", b.Noise, a.Noise, higherIsBetter: false, "0.0");
            Row(html, "Blur (pixels)", b.Blur, a.Blur, higherIsBetter: false, "0.0");
            Row(html, "Tonal range (contrast)", b.DynamicRange, a.DynamicRange, higherIsBetter: true, "0");
            html.Append("</tbody></table>");
        }

        // Composite image: what happened to each panel
        if (job.Panels.Count > 0)
        {
            html.Append("<table><thead><tr><th>Panel</th><th colspan=\"3\">Treatment</th></tr></thead><tbody>");
            foreach (var panel in job.Panels)
            {
                var text = Wording.Panel(panel);
                Fact(html, text[..text.IndexOf(':')], text[(text.IndexOf(':') + 2)..]
                    + (panel.Redraw is { Redrawn: true, InChIKey: { } key } ? " · InChIKey " + key : ""));
            }

            html.Append("</tbody></table>");
        }

        // Redraw: the chemical proof (or why the image was not redrawn)
        if (job.Redraw is { } redraw && Wording.HasChemicalDetails(redraw, job.ContentKind) && job.Panels.Count == 0)
        {
            if (redraw.Redrawn)
            {
                html.Append("<table><thead><tr><th>")
                    .Append(redraw.Proven ? "Redrawn structure (verified by the automatic checks: a chemist must confirm it matches the original)" : "Redrawn structure (NOT verified: compare every atom and bond)")
                    .Append("</th><th colspan=\"3\">Result</th></tr></thead><tbody>");
                Fact(html, "Molecular formula", redraw.Formula ?? "—");
                Fact(html, "Molecular weight", redraw.MolecularWeight is { } mw ? mw.ToString("0.00", CultureInfo.InvariantCulture) + " g/mol" : "—");
                Fact(html, "Stereocentres", redraw.Stereocentres.Count > 0 ? string.Join(", ", redraw.Stereocentres) : "none");
                Fact(html, "InChIKey", redraw.InChIKey ?? "—");
                Fact(html, "SMILES", redraw.Smiles ?? "—");
                if (redraw.Proven)
                {
                    Fact(html, "Checks passed", "bonds, wedges and labels in the picture · OCR labels · second reading · redraw read back identical");
                }
                else
                {
                    Fact(html, "Not verified because", string.Join("; ", redraw.Unproven ?? []));
                }

                if (redraw.Layout is { Length: > 0 } layout)
                {
                    Fact(html, "Drawing", layout);
                }
                if (redraw.Note is { Length: > 0 } note)
                {
                    Fact(html, "Note", note);
                }

                html.Append("</tbody></table>");
            }
            else
            {
                html.Append("<p class=\"muted\"><b>Not redrawn.</b> The image was improved instead.</p>");
            }
        }

        // Accuracy
        // Chemical structure check: exactly what did not match (the same numbered list as in the app)
        if (Wording.ShowsChemistry(job) && Wording.Mismatches(job) is { Count: > 0 } mismatches)
        {
            html.Append("<p class=\"note\"><b>").Append(job.IsRedrawn ? "What does not match (redraw not verified):" : "Why it was not redrawn:")
                .Append("</b></p><ol class=\"mismatches\">");
            foreach (var item in mismatches)
            {
                html.Append("<li>").Append(E(item.Message)).Append("</li>");
            }

            html.Append("</ol>");
        }

        if (job.Fidelity is { } f && !job.IsRedrawn)
        {
            html.Append("<table><thead><tr><th>Accuracy (compared with the original)</th><th colspan=\"3\">Result</th></tr></thead><tbody>");
            Fact(html, "Edges kept (lines, letters, shapes)", Pct(f.EdgesKept));
            Fact(html, "New edges (possible artefacts)", Pct(f.NewEdges));
            Fact(html, "Structural similarity (SSIM)", Pct(f.Ssim));
            Fact(html, "Brightness change", f.BrightnessShift.ToString("0.0", CultureInfo.InvariantCulture) + " of 255");
            if (Wording.TextCheck(job.TextCheck) is { Length: > 0 } text)
            {
                Fact(html, "Text check (OCR before and after)", text);
            }

            if (Wording.Choice(job.Choice) is { Length: > 0 } choice)
            {
                Fact(html, "Version kept (AI repair)", choice);
            }

            if (Wording.Devices(job.AiDevices) is { Length: > 0 } devices)
            {
                Fact(html, "AI device", devices);
            }
            html.Append("</tbody></table>");
        }

        if (job.NeedsReview && Wording.ReviewReasons(job).ToList() is { Count: > 0 } reviewReasons)
        {
            html.Append("<p class=\"note\"><b>Review:</b> ").Append(E(string.Join("; ", reviewReasons.Select(Wording.ReviewReason)))).Append("</p>");
        }

        if (job.Notes.Count > 0)
        {
            html.Append("<p class=\"muted\">").Append(E(string.Join("; ", job.Notes))).Append("</p>");
        }

        html.Append("</div></article>");
    }

    private static void Tile(StringBuilder html, string value, string label) =>
        html.Append("<div class=\"tile\"><b>").Append(E(value)).Append("</b><span>").Append(E(label)).Append("</span></div>");

    private static void Thumb(StringBuilder html, string? relative, string root, string caption)
    {
        html.Append("<figure>");
        var path = relative is null ? null : Path.Combine(root, relative);
        if (path is not null && File.Exists(path))
        {
            html.Append("<img alt=\"").Append(caption).Append("\" src=\"data:image/jpeg;base64,")
                .Append(Convert.ToBase64String(File.ReadAllBytes(path))).Append("\">");
        }
        else
        {
            html.Append("<div class=\"empty\">—</div>");
        }

        html.Append("<figcaption>").Append(caption).Append("</figcaption></figure>");
    }

    private static void Row(StringBuilder html, string label, double before, double after, bool higherIsBetter, string format)
    {
        var better = higherIsBetter ? after > before * 1.02 : after < before * 0.98;
        var worse = higherIsBetter ? after < before * 0.98 : after > before * 1.02;
        var mark = better ? "<span class=\"up\">improved</span>" : worse ? "<span class=\"down\">lower</span>" : "<span class=\"muted\">same</span>";
        html.Append("<tr><td>").Append(label).Append("</td><td class=\"num\">").Append(before.ToString(format, CultureInfo.InvariantCulture))
            .Append("</td><td class=\"num\">").Append(after.ToString(format, CultureInfo.InvariantCulture)).Append("</td><td>").Append(mark).Append("</td></tr>");
    }

    private static void Fact(StringBuilder html, string label, string value) =>
        html.Append("<tr><td>").Append(label).Append("</td><td class=\"num\" colspan=\"3\">").Append(E(value)).Append("</td></tr>");

    private static string Pct(double value) => (value * 100).ToString("0.0", CultureInfo.InvariantCulture) + " %";

    private static string E(string? text) => WebUtility.HtmlEncode(text ?? "");

    private static string? Capitalise(string? text) =>
        string.IsNullOrEmpty(text) ? text : char.ToUpperInvariant(text[0]) + text[1..];

    private const string Css = """
        :root{--bg:#f4f7fa;--card:#fff;--fg:#17212b;--muted:#5a6775;--line:#d8dfe6;--accent:#1d5c8c;--good:#1f7a4d;--goodbg:#e2f3ea;--warn:#9a5b00;--warnbg:#fbefd9;--high:#155e3c;--highbg:#d3efe0}
        @media (prefers-color-scheme:dark){:root{--bg:#10161c;--card:#18212a;--fg:#e4ebf2;--muted:#9aa8b6;--line:#2b3845;--accent:#79b4e3;--good:#6fcf9c;--goodbg:#15352a;--warn:#e9b562;--warnbg:#3a2c12;--high:#8be0b4;--highbg:#123826}}
        *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 "Segoe UI",system-ui,sans-serif}
        main{max-width:1000px;margin:0 auto;padding:32px 18px 64px;display:grid;gap:22px}
        h1{font-size:28px;margin:4px 0}h2{font-size:20px;margin:8px 0 0}h3{font-size:16px;margin:0}
        .eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:var(--accent);margin:0}
        .muted{color:var(--muted)}p{margin:4px 0}
        .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
        .tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:grid}
        .tile b{font-size:26px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:13px}
        .explain{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
        .explain ul{padding-left:20px;margin:8px 0;display:grid;gap:6px}
        .image{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr);gap:16px}
        @media (max-width:720px){.image{grid-template-columns:1fr}}
        .thumbs{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-content:start}
        figure{margin:0;display:grid;gap:4px}figure img,.empty{width:100%;aspect-ratio:4/3;object-fit:contain;background:var(--bg);border-radius:6px}
        .empty{display:grid;place-items:center;color:var(--muted)}figcaption{font-size:12px;color:var(--muted);text-align:center}
        .body{min-width:0;display:grid;gap:6px;align-content:start}.title{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
        .pill{display:inline-block;font-size:12px;font-weight:600;padding:3px 9px;border-radius:999px;background:var(--bg);color:var(--muted)}
        .pill.good{background:var(--goodbg);color:var(--good)}.pill.high{background:var(--highbg);color:var(--high)}.pill.verified{background:var(--accent);color:var(--card)}.pill.check{background:var(--warnbg);color:var(--warn)}
        table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:4px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
        th{font-size:12px;color:var(--muted);font-weight:600}.num{font-variant-numeric:tabular-nums}
        .steps{margin:0;padding-left:20px;display:grid;gap:3px}.up{color:var(--good);font-weight:600}.down{color:var(--warn);font-weight:600}
        .note{background:var(--warnbg);color:var(--warn);border-radius:8px;padding:8px 10px}
        """;
}
