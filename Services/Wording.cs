using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImageQualityEnhancer.Models;

namespace ImageQualityEnhancer.Services;

/// <summary>Plain-language names for image locations, processing steps and review reasons (screens and report).</summary>
public static class Wording
{
    public static string Location(ImageJob job)
    {
        var location = job.Location;
        switch (job.SourceType)
        {
            case SourceType.Pdf when location.Placements is { Count: > 0 } places:
                var pages = places.Select(p => p.PageIndex + 1).Distinct().ToList();
                var where = pages.Count == 1 ? $"Page {pages[0]}" : $"Pages {string.Join(", ", pages.Take(4))}{(pages.Count > 4 ? "…" : "")}";
                return places.Count > 1 ? $"{where} (shown {places.Count}×)" : where;
            case SourceType.Docx:
                var part = Path.GetFileNameWithoutExtension(location.PartUri ?? "");
                var area = part.StartsWith("header") ? ("Header " + part[6..]).TrimEnd()
                    : part.StartsWith("footer") ? ("Footer " + part[6..]).TrimEnd()
                    : part == "document" ? "Body" : part;
                return $"{area} image {location.Ordinal + 1}{(location.InTable ? " · in table" : "")}{(location.Anchored ? " · floating" : "")}";
            case SourceType.Image:
                var name = Path.GetFileName(location.SourcePath ?? "");
                return location.FrameIndex > 0 ? $"{name} (page {location.FrameIndex + 1})" : name;
            default:
                return job.Id;
        }
    }

    public static string Operation(string op) => op switch
    {
        "denoise" => "noise reduction",
        "contrast" => "contrast",
        "local_contrast" => "clarity",
        "sharpen" => "sharpening",
        "resize" => "more pixels",
        "fine_grid" => "fine-detail processing",
        "smooth_speckle" => "speckle smoothing",
        "clean_background" => "background cleaning",
        "deblur" => "deblurring",
        "crisp_text" => "crisp text",
        "regional" => "region-by-region cleaning",
        "bolden" => "bolder text",
        "redraw" => "redrawn chemical structure",
        "ai_photo" => "AI clarity",
        "panels" => "panel by panel",
        "remove_specks" => "speck removal",
        "ai_grain" => "AI grain removal",
        "ai_jpeg" => "AI JPEG repair",
        "ai_deblur_text" => "AI text de-blur",
        _ => op.Replace('_', ' '),
    };

    /// <summary>What each step does, for the report.</summary>
    public static string OperationExplained(string op) => op switch
    {
        "denoise" => "Removes random grain while keeping real texture and edges.",
        "contrast" => "Spreads the tones so faded images regain contrast. A faded photo also loses its colour cast (each colour gets its own black and white point).",
        "local_contrast" => "Brings out detail in flat-looking areas (photos, Strong and Maximum levels). It changes the look of the photo slightly.",
        "sharpen" => "Makes soft edges crisper.",
        "resize" => "Adds pixels (1.5× or 2×) so the image stays sharp when viewed or printed; its size on the page is unchanged.",
        "smooth_speckle" => "Fades photocopy grain and scanner speckle on paper areas; grid lines and table rules are kept.",
        "clean_background" => "Evens out shading and turns grey or yellowed paper white; ink, coloured labels, chart bars and grid lines are kept.",
        "deblur" => "Measures the blur and reverses it (deconvolution).",
        "crisp_text" => "Gives text and lines clean edges; ink is only darkened, never lightened, and colours are kept.",
        "regional" => "Cleans document areas and treats photo areas as photos on mixed pages.",
        "fine_grid" => "The following steps work on a 2× finer grid inside the program, then the result is reduced back to exactly the original pixel size, for smoother, cleaner edges. The image size does not change.",
        "panels" => "The image is a sheet of framed panels: each panel was found and processed as an image of its own (structures, photos, tables and text each get their own treatment); titles and frames are kept as they are.",
        "remove_specks" => "Removes isolated dark specks (photocopy and scanner dirt) far from any text or line; dots near letters, such as full stops and decimal points, are always kept.",
        "ai_grain" => "An AI model (SCUNet) removes scan and camera grain at the same pixel size. It was kept because it measured better than the filters alone: as readable (text) or as detailed (pictures), with less grain.",
        "ai_jpeg" => "An AI model (FBCNN) removes JPEG blocks and ringing from a strongly compressed image at the same pixel size. Used only for JPEGs saved at low quality, and kept only when it measured better.",
        "ai_deblur_text" => "An AI model (DocRes) sharpens blurred document text. Kept only when every letter and digit it shows was already readable in the original, so no character can be invented.",
        "ai_photo" => "An AI model (Real-ESRGAN) removes noise, compression blocks and blur from the photo at 4× and the result is reduced back to the original pixel size. Photos only; it can add fine texture, so compare it zoomed in.",
        "redraw" => "Read as a chemical structure, checked against the picture (every bond, wedge and label) and by OCR, read a second time, and redrawn cleanly with RDKit in the original layout; the redraw reads back as the same molecule that was read (same InChIKey, including stereochemistry). These checks do not prove that it matches the original: a chemist compares and approves it.",
        "bolden" => "Makes thin, faint text and lines slightly thicker and darker in their own colour; letter holes, double bonds and close lines stay open.",
        _ => "",
    };

    /// <summary>"Text check: passed · 214 characters read the same before and after", or what changed.</summary>
    public static string TextCheck(TextCheckResult? check)
    {
        if (check is null)
        {
            return "";
        }

        if (!check.TextFound)
        {
            return "Text check: no text found";
        }

        if (!check.Flag)
        {
            var same = check.Characters - check.Changed;
            var passed = $"Text check: passed · {same:N0} of {check.Characters:N0} characters read the same before and after";
            return check.NumbersUncertain.Count > 0
                ? passed + " · a symbol reads differently but its strokes are intact: " + string.Join(", ", check.NumbersUncertain)
                : passed;
        }

        var parts = new List<string> { $"Text check: {check.Changed:N0} of {check.Characters:N0} characters read differently" };
        if (check.NumbersChanged.Count > 0)
        {
            parts.Add("numbers: " + string.Join(", ", check.NumbersChanged));
        }
        else if (check.Examples.Count > 0)
        {
            parts.Add("e.g. " + string.Join(", ", check.Examples.Take(3)));
        }

        return string.Join(" · ", parts);
    }

    /// <summary>"Redrawn: C5H11NO2S · 149.21 g/mol · stereocentres C4 S · InChIKey …", or why it was not redrawn.</summary>
    /// <summary>Names of the versions AI repair compares.</summary>
    public static string VersionName(string name) => name switch
    {
        "filters" => "the filters",
        "original" => "the original",
        "ai_clean" => "AI repair",
        "ai_deblur" => "AI text de-blur",
        _ => name.Replace('_', ' '),
    };

    /// <summary>
    /// "Best of 2 versions: AI repair kept (as readable, less grain). Not used: AI text de-blur (shows digits the
    /// original does not show readably)." Empty when no alternative was made.
    /// </summary>
    public static string Choice(VersionChoice? choice)
    {
        if (choice is null || choice.Candidates.Count < 2)
        {
            return "";
        }

        var kept = choice.Candidates.FirstOrDefault(c => c.Name == choice.Chosen);
        var others = choice.Candidates.Where(c => c.Name != choice.Chosen).ToList();
        // Compared only with the versions that could have been used (rejected ones are listed separately).
        var usable = others.Where(o => o.Rejected is null).ToList();
        var why = kept is null ? "" : kept.Legibility is { } legibility
            ? (usable.Any(o => o.Legibility is { } l && legibility > l * 1.02) ? " (more readable)" : " (as readable, least grain)")
            : " (least grain, detail kept)";
        var text = $"Best of {choice.Candidates.Count} versions: {VersionName(choice.Chosen)} kept{why}.";
        var rejected = others.Where(o => o.Rejected is not null).Select(o => $"{VersionName(o.Name)} ({o.Rejected})").ToList();
        if (rejected.Count > 0)
        {
            text += " Not used: " + string.Join("; ", rejected) + ".";
        }

        if (choice.LessReadable)
        {
            text += " The text reads less clearly than in the original.";
        }

        return text;
    }

    /// <summary>"AI ran on: SCUNet – processor, FBCNN – NVIDIA GeForce RTX 3050 Laptop GPU". Empty without AI.</summary>
    public static string Devices(IReadOnlyDictionary<string, string> devices) =>
        devices.Count == 0 ? "" : "AI ran on: " + string.Join(", ", devices.Select(d => $"{d.Key} – {d.Value}"));

    // A negative chemical detection is normal for tables/photos, not a structure error. Never hide a warning
    // about a picture that was actually redrawn, including old saved results that need manual review.
    // Legacy confidence-only guesses are not positive chemical detection; retain legacy molecular identities.
    /// <summary>
    /// Chemical details (structure checks, redraw outcome, chemical errors) belong to chemical pictures only: tables,
    /// graphs, photos and other images never show them. A picture that was actually redrawn always shows them, so a
    /// redraw is never left without its warning.
    /// </summary>
    public static bool ShowsChemistry(ImageJob job) =>
        job.IsChemicalCategory || job.IsRedrawn || job.Panels.Any(p => p.Redraw is { Redrawn: true })
        // Jobs saved before pictures were classified keep the earlier rule (a reading with a molecular identity).
        || job.Classification is null && job.ManualCategory is null
           && (HasChemicalDetails(job.Redraw, job.ContentKind) || job.Panels.Any(p => HasChemicalDetails(p.Redraw, p.Content)));

    public static bool HasChemicalDetails(RedrawResult? redraw, string? contentKind = null) =>
        redraw is not null && (redraw.Redrawn || contentKind != "photo" &&
            (redraw.IsChemical == true || redraw.IsChemical is null && !string.IsNullOrWhiteSpace(redraw.InChIKey)));

    public static IEnumerable<string> ReviewReasons(ImageJob job) => job.ReviewReasons.Where(reason => reason switch
    {
        "redrawn_structure" => job.IsRedrawn || job.Panels.Any(p => p.Redraw is { Redrawn: true }),
        "unproven_redraw" => job.IsUnprovenRedraw || job.Panels.Any(p => p.Redraw is { Redrawn: true, Proven: false }),
        _ => true,
    });

    /// <summary>
    /// The mismatches of a chemical drawing, one entry per distinct sentence (a sentence that applies in several
    /// places is listed once), numbered like the marks on the picture. Panels of a sheet are prefixed "Panel n:".
    /// </summary>
    public static IReadOnlyList<(int Number, string Message, IReadOnlyList<double[]> Boxes)> Mismatches(ImageJob job)
    {
        var all = new List<(string Message, double[]? Box)>();

        // Older worker results, and a few checks without a location in the picture, have only a plain reason
        // rather than a visual finding. Those reasons are still the exact reason an "Redraw all" result is not
        // proven, so show them in the review screen instead of leaving the user with a generic warning.
        static IEnumerable<(string Message, double[]? Box)> Entries(RedrawResult? redraw, string? contentKind)
        {
            if (redraw is null || !HasChemicalDetails(redraw, contentKind) || (redraw.Redrawn && redraw.Proven))
            {
                return Enumerable.Empty<(string Message, double[]? Box)>();
            }

            if (redraw.Findings is { Count: > 0 })
            {
                return redraw.Findings.Select(f => (f.Message, f.Box));
            }

            var reasons = redraw.Redrawn ? redraw.Unproven : redraw.Reasons;
            return (reasons ?? []).Where(reason => !string.IsNullOrWhiteSpace(reason))
                .Select(reason => (reason, (double[]?)null));
        }

        if (job.Panels.Count > 0)
        {
            foreach (var panel in job.Panels)
            {
                foreach (var f in Entries(panel.Redraw, panel.Content))
                {
                    all.Add(($"Panel {panel.Number}: {f.Message}", f.Box));
                }
            }
        }
        else if (job.Redraw is { } redraw && (!redraw.Redrawn || !redraw.Proven))
        {
            all.AddRange(Entries(redraw, job.ContentKind));
        }

        return all.GroupBy(f => f.Message)
            .Select((g, k) => (k + 1, g.Key, (IReadOnlyList<double[]>)g.Where(f => f.Box is not null).Select(f => f.Box!).ToList()))
            .ToList();
    }

    public static string Redraw(RedrawResult? redraw, string? contentKind = null)
    {
        if (redraw is null || !HasChemicalDetails(redraw, contentKind))
        {
            return "";
        }

        if (!redraw.Redrawn)
        {
            return "Not redrawn: " + (redraw.Reasons.Count > 0 ? redraw.Reasons[0] : "the reading could not be verified") + " (enhanced instead)";
        }

        var parts = new List<string> { (redraw.Proven ? "Redrawn (verified by the automatic checks, not by a chemist yet): " : "Redrawn, NOT VERIFIED: ") + redraw.Formula };
        if (redraw.MolecularWeight is { } mw)
        {
            parts.Add($"{mw:0.00} g/mol");
        }

        if (redraw.Stereocentres.Count > 0)
        {
            parts.Add("stereocentres " + string.Join(", ", redraw.Stereocentres));
        }

        parts.Add("InChIKey " + redraw.InChIKey);
        var text = string.Join(" · ", parts);
        if (!redraw.Proven && redraw.Unproven is { Count: > 0 } failed)
        {
            text += ". Not verified because: " + string.Join("; ", failed.Take(2))
                    + (failed.Count > 2 ? $" (and {failed.Count - 2} more)" : "") + ". Compare every atom and bond with the original.";
        }

        if (redraw.Layout is { Length: > 0 } layout)
        {
            text += " Drawn " + (layout.StartsWith("drawn ") ? layout[6..] : layout) + ".";
        }

        return text;
    }

    /// <summary>"Panel 1 (text / diagram): redrawn C6H6", "Panel 4 (photo): AI clarity, contrast", ...</summary>
    public static string Panel(PanelResult panel)
    {
        var what = panel.Redraw is { Redrawn: true } r
            ? $"redrawn {r.Formula}" + (r.Stereocentres.Count > 0 ? $" ({string.Join(", ", r.Stereocentres)})" : "")
              + (r.Proven ? "" : ", not verified")
            : panel.Operations.Count == 0
                ? "already good, unchanged"
                : string.Join(", ", panel.Operations.Where(o => o is not ("fine_grid" or "regional")).Select(Operation).Distinct());
        var note = panel.Redraw is { Redrawn: false } nr && HasChemicalDetails(nr) && nr.Reasons.Count > 0 && panel.Content == "text_or_diagram"
                   && nr.Reasons[0].Contains("enough confidence") is false
            ? " · not redrawn: " + nr.Reasons[0]
            : "";
        return $"Panel {panel.Number} ({ContentKind(panel.Content).ToLowerInvariant()}): {what}{note}";
    }

    /// <summary>Enhancement levels in order (worker strength names).</summary>
    public static IReadOnlyList<string> LevelKeys { get; } = ["gentle", "standard", "strong", "maximum"];

    /// <summary>Short level name: "Light", "Standard", "Strong" or "Maximum".</summary>
    public static string LevelName(string? key) => key switch
    {
        "gentle" => "Light",
        "strong" => "Strong",
        "maximum" => "Maximum",
        _ => "Standard",
    };

    public static string LevelKey(string? name) =>
        LevelKeys.FirstOrDefault(k => LevelName(k).Equals(name, System.StringComparison.OrdinalIgnoreCase)) ?? "standard";

    /// <summary>"Level 3 · Strong".</summary>
    public static string Level(string? key)
    {
        var index = LevelKeys.ToList().IndexOf(key ?? "standard");
        return $"Level {(index < 0 ? 2 : index + 1)} · {LevelName(key)}";
    }

    /// <summary>Choices for bolder text, in the order of <see cref="BoldValue"/>.</summary>
    public static IReadOnlyList<string> BoldChoices { get; } = ["Automatic (levels 3 and 4)", "On", "Off"];

    public static bool? BoldValue(string? choice) => choice == "On" ? true : choice == "Off" ? false : null;

    public static string BoldChoice(bool? value) => value switch { true => "On", false => "Off", _ => BoldChoices[0] };

    public static string ReviewReason(string reason) => reason switch
    {
        "text_or_diagram" => "text or diagram: check that every letter, number and line is still right",
        "large_change" => "the picture changed a lot: compare it closely with the original",
        "possible_artifacts" => "new marks may have appeared (light halos or extra edges): zoom in to check",
        "possible_detail_loss" => "some fine detail may have faded: zoom in to check",
        "brightness_shift" => "the picture became lighter or darker overall: check that nothing faded",
        "color_conversion" => "colours were converted from print colours (CMYK) to screen colours: check them",
        "ai_deblurred_text" => "AI sharpened blurred text: every letter and digit was checked against the original, but compare them before approving",
        "text_less_readable" => "the text reads less clearly than in the original: keeping the original may be better",
        "ai_restored" => "AI-restored photo: fine texture may be added by the model; compare it zoomed in",
        "unproven_redraw" => "redrawn but not verified: compare every atom, bond and stereo mark with the original before approving",
        "redrawn_structure" => "redrawn chemical structure: a chemist must compare it with the original before approving",
        "text_changed" => "the text reader reads some letters or numbers differently after the change: compare them with the original",
        "heavy_blur" => "the original is very blurred: some letters may not have been recovered correctly, check them against the source",
        _ => reason,
    };

    /// <summary>
    /// Why a picture was kept as it is, or could not be processed, in plain words with what (if anything) to do.
    /// </summary>
    public static string KeptOrFailed(ImageJob job)
    {
        var detail = string.IsNullOrWhiteSpace(job.Error) ? "" : " (" + job.Error.TrimEnd('.') + ")";
        return job.Reason switch
        {
            "already_good" => "Already clear. Nothing needed fixing, so the original is kept exactly as it is.",
            "vector" => "This is a drawing made of lines and shapes (a vector graphic), not of pixels. It stays sharp at any size, so it is kept as it is.",
            "unsupported_format" => "The tool cannot open this picture format, so the original is kept. Save it as PNG or JPEG to enhance it.",
            "animated" => "Animated picture: kept as it is, so the animation is not lost.",
            "disabled" => "Enhancement was switched off for this picture.",
            "bilevel" => "Pure black-and-white picture: there is no grey or colour to clean, so it is kept as it is.",
            "high_bit_depth" => "High-precision (16-bit) picture: kept as it is, so none of its extra detail is lost.",
            "too_large" => "Too large to process safely on this computer, so the original is kept" + detail + ".",
            "too_small" => "Too small to benefit from enhancement (an icon or a tiny mark): kept as it is.",
            "unsupported_colorspace" => "Uses print colours (CMYK or similar) that cannot be enhanced safely: kept as it is.",
            "inline_image" => "Stored inside the page's drawing commands, where it cannot be replaced safely: kept as it is.",
            "ole_preview" => "Preview of an embedded object (for example a ChemDraw structure). Its own program redraws it, so it is kept as it is.",
            "svg_fallback" => "Backup copy of an SVG picture that Word does not show: kept as it is.",
            "linked_image" => "The picture is linked from outside the document, not stored in it: kept as it is.",
            "missing_image" => "The picture is missing from the document, so there is nothing to enhance.",
            "decode_failed" or "read_failed" => "The picture's data is damaged and could not be opened. The original stays in the document unchanged.",
            "worker_crashed" => "The image processor stopped twice while working on this picture, so the original is kept. Try another strength, or keep the original.",
            "extract_failed" => "The picture could not be taken out of the document. The original stays in the document unchanged.",
            "encode_failed" or "write_failed" => "The enhanced picture could not be saved, so the original is kept.",
            _ => Sentence(job.Error ?? "this picture was kept unchanged"),
        };
    }

    private static string Sentence(string text) => char.ToUpperInvariant(text[0]) + text[1..] + (text.EndsWith('.') ? "" : ".");

    /// <summary>
    /// What optional AI enhancement may do for a picture, before it is tried: one line per suitable model ("Grain
    /// detected — SCUNet may help."), whether it is installed, or why AI is not offered. Never a promise.
    /// </summary>
    public static string AiSuggestions(ImageJob job)
    {
        if (job.IsRedrawn)
        {
            return "AI repair is not offered for a redrawn structure.";
        }

        if (job.Operations.Contains("panels"))
        {
            return "AI repair is not offered for a sheet of several pictures (each panel was processed on its own).";
        }

        if (job.AiSuggestions.Count == 0)
        {
            return "No suitable AI repair detected.";
        }

        return string.Join("\n", job.AiSuggestions.Select(s => s.Installed
            ? s.Text
            : $"{s.Text} {s.Model} is not installed, so it cannot be tried (run worker/setup_ai.py)."));
    }

    /// <summary>
    /// The AI outcome in a few words, for the panel's headline: "AI enhancement applied (SCUNet)", "AI version ready
    /// (SCUNet) · not in use", "AI enhancement not applied (SCUNet)" or "AI enhancement could not run".
    /// </summary>
    public static string AiHeadline(ImageJob job) => job.Ai switch
    {
        null => "",
        { Accepted: true } ai when job.UseAiVersion => $"AI enhancement applied ({ai.ModelNames})",
        { Accepted: true } ai => $"AI version ready ({ai.ModelNames}) · not in use",
        { Status: "rejected" } ai => $"AI enhancement not applied ({ai.ModelNames})",
        { Status: "not_eligible" } => "AI enhancement not applied",
        _ => "AI enhancement could not run",
    };

    /// <summary>What happened when AI was tried and what the user can do now, under the headline.</summary>
    public static string AiOutcome(ImageJob job)
    {
        if (job.Ai is not { } ai)
        {
            return "";
        }

        var normal = NormalVersion(job);
        switch (ai.Status)
        {
            case "accepted" when job.UseAiVersion:
                return $"The AI version is now used for this picture. It passed the {AiChecks(ai)}. AI may not be better for every " +
                       $"picture: compare it with the {normal}, and switch back if you prefer that. It still needs your approval.";
            case "accepted":
                var why = job.Approval switch
                {
                    ApprovalStatus.Approved => $"You had already approved the {normal}, so it stays in use until you choose the AI version (you will be asked first).",
                    ApprovalStatus.KeepOriginal => "You chose to keep the original, so nothing changes unless you choose the AI version.",
                    _ => $"The {normal} is used. Choose the AI version to use it instead.",
                };
                return $"It passed the {AiChecks(ai)}. {why}";
            case "rejected":
                return $"It did not pass the checks: {AiReason(ai.Reasons.FirstOrDefault())}. The {normal} stays in use.";
            case "not_eligible":
                return $"No suitable AI repair was detected for this picture. The {normal} stays in use.";
            case "unavailable":
                return string.Join(" ", ai.Reasons) + $" The {normal} stays in use.";
            default:
                return $"{Sentence(AiReason(ai.Reasons.FirstOrDefault()))} The {normal} stays in use.";
        }
    }

    /// <summary>The outcome in one line for the report and notices: the headline and the deciding fact.</summary>
    public static string AiSummary(ImageJob job)
    {
        if (job.Ai is not { } ai)
        {
            return "";
        }

        var headline = AiHeadline(job);
        return ai.Status switch
        {
            "accepted" => $"{headline}. It passed the {AiChecks(ai)}.",
            "rejected" => $"{headline}: {AiReason(ai.Reasons.FirstOrDefault())}.",
            "not_eligible" => $"{headline}: no suitable AI repair detected.",
            _ => $"{headline}: {string.Join(" ", ai.Reasons.Select(AiReason)).TrimEnd('.')}.",
        };
    }

    private static string NormalVersion(ImageJob job) => job.Status == ProcessingStatus.Completed ? "OpenCV result" : "original";

    private static string AiChecks(AiResult ai)
    {
        var checks = new List<string>();
        if (ai.TextPassed == true)
        {
            checks.Add("text check");
        }

        checks.Add("detail check");
        if (ai.ChemicalNote is not null)
        {
            checks.Add("chemical check");
        }

        return JoinAnd(checks);
    }

    /// <summary>A check's reason in plain words, without the model's name in front.</summary>
    private static string AiReason(string? reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
        {
            return "it did not pass the checks";
        }

        var text = reason.Contains(": ") ? reason[(reason.IndexOf(": ", System.StringComparison.Ordinal) + 2)..] : reason;
        return text switch
        {
            "more detail lost or added than with the filters" => "it loses or adds more detail than the OpenCV result",
            "shows digits the original does not show readably" => "it shows digits that are not clear in the original, so they may be guessed",
            "shows letters the original does not show readably" => "it shows letters that are not clear in the original, so they may be guessed",
            "a number reads differently from the original" => "a number reads differently from the original",
            "too many characters read differently from the original" => "too many letters read differently from the original",
            "the original has no readable text to check the sharpened letters against" =>
                "the original has no readable text, so the sharpened letters could not be checked",
            _ => text,
        };
    }

    private static string JoinAnd(IReadOnlyList<string> items) =>
        items.Count <= 1 ? string.Concat(items) : string.Join(", ", items.Take(items.Count - 1)) + " and " + items[^1];

    /// <summary>"Chemical diagram", "Flowchart or graph", "Table", "Other image" or "Not classified".</summary>
    public static string CategoryName(ImageCategory category) => category switch
    {
        ImageCategory.Chemical => "Chemical diagram",
        ImageCategory.FlowchartOrGraph => "Flowchart or graph",
        ImageCategory.Table => "Table",
        ImageCategory.Other => "Other image",
        _ => "Not classified",
    };

    /// <summary>The group a picture is listed under, for its row: "Table · 99 %", "Chemical diagram · set by you",
    /// "Sheet: chemical diagram, table, other image".</summary>
    public static string CategoryLine(ImageJob job)
    {
        if (job.ManualCategory is { } manual)
        {
            return CategoryName(manual) + " · set by you";
        }

        if (job.Classification is not { } c)
        {
            return "Not classified";
        }

        if (c.Panels.Count > 0)
        {
            return "Sheet: " + string.Join(", ", job.Groups.Select(g => CategoryName(g).ToLowerInvariant()));
        }

        return c.Confidence is { } p ? $"{CategoryName(c.Category)} · {p * 100:0} %" + (c.Uncertain ? " (unsure)" : "") : CategoryName(c.Category);
    }

    /// <summary>
    /// Whether the user should be asked about the picture's group: the classifier is unsure (for the picture or one of
    /// its panels), it looks chemical but could not be read, it could not be classified, or the user already changed it.
    /// Confident suggestions are not offered for change.
    /// </summary>
    public static bool IsCategoryDoubtful(ImageJob job) =>
        job.ManualCategory is not null || job.Classification is not { Confidence: not null } c
        || c.Uncertain || c.Panels.Any(p => p.Uncertain) || job.IsPossibleChemical;

    /// <summary>The question shown with the category chooser of a doubtful picture.</summary>
    public static string CategoryQuestion(ImageJob job)
    {
        if (job.ManualCategory is { } manual)
        {
            return $"You set this picture's group to {CategoryName(manual).ToLowerInvariant()}. Change it here, or go back to the suggestion.";
        }

        if (job.Classification is not { Confidence: { } p } c)
        {
            return "This picture could not be sorted into a group automatically. Which kind of picture is it?";
        }

        if (job.IsPossibleChemical)
        {
            return "This looks like a chemical structure, but its atoms and bonds could not be read reliably. " +
                   "If it is not a chemical structure, choose the right group.";
        }

        var unsurePanel = c.Panels.FirstOrDefault(x => x.Uncertain);
        return unsurePanel is not null && !c.Uncertain
            ? $"Panel {unsurePanel.Number} of this sheet is hard to recognise. Check the picture's group:"
            : $"Not sure what kind of picture this is: it looks most like {CategoryNoun(c.Category)} " +
              $"({p * 100:0} % sure). Which kind of picture is it?";
    }

    private static string CategoryNoun(ImageCategory category) => category switch
    {
        ImageCategory.Chemical => "a chemical diagram",
        ImageCategory.FlowchartOrGraph => "a flowchart or graph",
        ImageCategory.Table => "a table",
        _ => "a photo or other image",
    };

    /// <summary>
    /// The classifier's suggestion in full: "Suggested by DocumentFigureClassifier v2.5: Table, 99 % sure (figure
    /// type: table)." plus the panels of a sheet, and the user's own choice when they made one.
    /// </summary>
    public static string ClassificationDetails(ImageJob job)
    {
        var lines = new List<string>();
        if (job.Classification is { } c)
        {
            var model = c.Model ?? "the picture classifier";
            lines.Add(c.Confidence is { } p
                ? $"Suggested by {model}: {CategoryName(c.Category)}, {p * 100:0} % sure" + (c.Label is { } l ? $" (most likely figure type: {l.Replace('_', ' ')})" : "")
                  + (c.Uncertain ? ". The classifier is unsure: check the category." : ".")
                : Sentence(c.Note ?? "not classified: the picture could not be read"));
            foreach (var panel in c.Panels)
            {
                lines.Add($"Panel {panel.Number}: {CategoryName(panel.Category)}, {panel.Confidence * 100:0} % sure" + (panel.Uncertain ? " (unsure)" : ""));
            }

            if (c.Panels.Count == 0 && c.Confidence is not null && c.Note is { Length: > 0 } note)
            {
                lines.Add(char.ToUpperInvariant(note[0]) + note[1..] + ".");
            }
        }
        else
        {
            lines.Add("Not classified (the picture classifier is not installed, or the picture has not been classified yet).");
        }

        if (job.ManualCategory is { } manual)
        {
            lines.Add($"Set by you: {CategoryName(manual)}. Your choice applies to the whole picture.");
        }

        return string.Join("\n", lines);
    }

    /// <summary>
    /// Where a chemical picture stands, keeping apart "classified as chemical" (the classifier's suggestion or the
    /// user's choice), MolScribe's reading (a second check) and "chemical structure verified" (every automatic redraw
    /// check passed). None of them proves that a reading or a redraw matches the original. Empty for other pictures.
    /// </summary>
    public static string ChemicalStatus(ImageJob job)
    {
        if (!job.IsChemicalCategory)
        {
            return "";
        }

        if (job.IsVerifiedStructure)
        {
            return "Chemical structure verified by the automatic checks: every bond, wedge and label was found in the picture, a second " +
                   "reading agreed, and the redraw reads back as the same molecule. Automatic checks do not prove that the redraw matches " +
                   "the original: a chemist must compare it and approve it.";
        }

        static string Reading(ChemicalCheck? check) => check switch
        {
            { IsRead: true } r => $"The structure reader (MolScribe) recognised {r.Atoms} atoms and {r.Bonds} bonds" + (r.Formula is { } f ? $" ({f})" : "") +
                                  ". This is a reading only, not a verified structure: compare it with the original.",
            { Status: "unavailable" } => "Possible chemical diagram — needs review. The structure reader (MolScribe) is not installed, " +
                                         "so its atoms and bonds were not read. A chemist should look at it.",
            { } u => "Possible chemical diagram — needs review. The structure reader (MolScribe) could not read it reliably" +
                     (u.Reasons.Count > 0 ? ": " + u.Reasons[0].Replace("MolScribe ", "it ") : "") + ". A chemist should look at it.",
            null => "Possible chemical diagram — needs review. Its atoms and bonds have not been read. A chemist should look at it.",
        };

        var lines = new List<string>
        {
            job.ManualCategory == ImageCategory.Chemical ? "Marked as a chemical diagram by you." : "Classified as a chemical diagram — not verified.",
        };
        var panels = job.ManualCategory is null ? job.Classification?.Panels.Where(p => p.Category == ImageCategory.Chemical).ToList() : null;
        if (panels is { Count: > 0 })
        {
            lines.AddRange(panels.Select(p => $"Panel {p.Number}: " + Reading(p.ChemicalCheck)));
        }
        else
        {
            lines.Add(Reading(job.WholeChemicalCheck));
        }

        return string.Join("\n", lines);
    }

    public static string ContentKind(string? kind) => kind switch
    {
        "photo" => "Photo",
        "text_or_diagram" => "Text / diagram",
        _ => "—",
    };
}
