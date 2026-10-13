using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using ImageQualityEnhancer.Models;
using ImageQualityEnhancer.Services;

namespace ImageQualityEnhancer.ViewModels;

/// <summary>A numbered box drawn on a picture, in the picture's own pixels (for example where a mismatch is).</summary>
public sealed record ImageMark(int Number, Avalonia.Rect Box);

/// <summary>One row of the gallery: a snapshot of an <see cref="ImageJob"/> plus display text and lazy thumbnails.</summary>
public sealed class ImageItemViewModel : ViewModelBase
{
    private readonly string _workspaceRoot;

    public ImageItemViewModel(ImageJob job, int index, string workspaceRoot)
    {
        Job = job;
        Index = index;
        _workspaceRoot = workspaceRoot;
    }

    public ImageJob Job { get; private set; }

    /// <summary>Position in the document order, used to keep filtered lists sorted.</summary>
    public int Index { get; }

    public string Id => Job.Id;

    public void Update(ImageJob job)
    {
        Job = job;
        OnPropertyChanged(string.Empty); // every displayed value may have changed
    }

    // ------------------------------------------------------------------ text

    public string Title => Wording.Location(Job);

    public string Details
    {
        get
        {
            var parts = new System.Collections.Generic.List<string>();
            if (Job.PixelSize is { } px)
            {
                parts.Add($"{px.Width}×{px.Height} px");
            }

            if (Job.DisplaySize is { } d)
            {
                parts.Add($"shown {d.Width:0}×{d.Height:0} pt");
            }

            if (Job.ContentKind is { } kind)
            {
                parts.Add(kind == "photo" ? "photo" : "text / diagram");
            }

            return string.Join(" · ", parts);
        }
    }

    public string StatusText => Job.Status switch
    {
        _ when Job.UseAiVersion && Job.Ai is { Accepted: true } => "AI enhanced · needs review",
        ProcessingStatus.Pending => "Waiting",
        ProcessingStatus.Processing => "Enhancing…",
        ProcessingStatus.Completed when Job.IsUnprovenRedraw => "Redrawn · not verified",
        ProcessingStatus.Completed when Job.IsRedrawn => "Redrawn · chemist review",
        ProcessingStatus.Completed when Job.Panels.Any(p => p.Redraw is { Redrawn: true }) => "Enhanced · structures to review",
        ProcessingStatus.Completed when Job.NeedsReview => "Enhanced · needs review",
        ProcessingStatus.Completed => "Enhanced",
        ProcessingStatus.Skipped when Job.Reason == "already_good" => "Already good · kept",
        ProcessingStatus.Skipped => "Kept unchanged",
        ProcessingStatus.Failed => "Failed · original kept",
        ProcessingStatus.Cancelled => "Cancelled",
        _ => Job.Status.ToString(),
    };

    public IBrush StatusBrush => Job.Status switch
    {
        _ when Job.UseAiVersion => AiBrush,
        ProcessingStatus.Completed when Job.NeedsReview => Brushes.DarkOrange,
        ProcessingStatus.Completed => Brushes.SeaGreen,
        ProcessingStatus.Failed => Brushes.Firebrick,
        ProcessingStatus.Processing => Brushes.SteelBlue,
        _ => Brushes.Gray,
    };

    /// <summary>Why the image was skipped or failed, or what to check when it needs review.</summary>
    public string? Explanation =>
        Job.Status is ProcessingStatus.Skipped or ProcessingStatus.Failed
            ? Wording.KeptOrFailed(Job)
            : Job.NeedsReview
                ? string.Join(Environment.NewLine, Wording.ReviewReasons(Job).Select(r => "• " + Capitalise(DescribeReview(r))))
                : null;

    public bool HasExplanation => !string.IsNullOrEmpty(Explanation);

    /// <summary>One actionable sentence on the main screen; technical reasons stay in Details.</summary>
    public string? ReviewSummary => Job.Status is ProcessingStatus.Skipped or ProcessingStatus.Failed && Job.IsPossibleChemical
        ? "Possible chemical diagram: it needs a chemist's review. " + Wording.KeptOrFailed(Job)
        : Job.Status is ProcessingStatus.Skipped or ProcessingStatus.Failed
        ? Wording.KeptOrFailed(Job)
        : Job.Status != ProcessingStatus.Completed ? null
        : Job.UseAiVersion && Job.Ai is { Accepted: true } chosenAi
            ? $"The AI version ({chosenAi.ModelNames}) is used. Compare it closely with the original before approving."
        : Job.IsUnprovenRedraw || Job.Panels.Any(p => p.Redraw is { Redrawn: true, Proven: false })
            ? "This redraw could not be fully verified. A chemist must compare it with the original before approving."
        : Job.IsRedrawn || Job.Panels.Any(p => p.Redraw is { Redrawn: true })
            ? "A chemist must compare the redrawn structure with the original before approving."
        : HasMismatches
            ? "A reliable redraw was not possible. The original drawing was enhanced instead. See Structure checks."
        : Job.IsPossibleChemical
            ? "Possible chemical diagram: it needs a chemist's review. Compare the result with the original before approving."
        : Job.TextCheck is { Flag: true } || Job.ReviewReasons.Any(r => r is "text_changed" or "text_less_readable" or "heavy_blur")
            ? "Check the text and numbers carefully. Keep the original if anything has changed."
        : Job.ContentKind == "photo"
            ? "Compare the photo with the original, then approve it or keep the original."
        : "Compare the result with the original, then approve it or keep the original.";

    public bool HasReviewSummary => !string.IsNullOrEmpty(ReviewSummary);

    public string ApprovalText => Job.Approval switch
    {
        ApprovalStatus.Approved => "✔ Approved",
        ApprovalStatus.KeepOriginal => "✕ Original kept",
        _ => CanApprove ? "To decide" : "",
    };

    public bool HasApprovalText => ApprovalText.Length > 0;

    public bool IsApproved => Job.Approval == ApprovalStatus.Approved;
    public bool CanApprove => ApprovalRules.CanApprove(Job);
    public bool CanReenhance => JobOrchestrator.CanReenhance(Job);

    public string OperationsText => Job.Operations.Count == 0
        ? ""
        : "Applied: " + string.Join(", ", Job.Operations.Select(Wording.Operation));

    /// <summary>How faithful the result is to the original, e.g. "Accuracy: High · 99.6 % of lines and letters kept".</summary>
    public string AccuracyText => Job.UseAiVersion && Job.Ai is { Fidelity: { } af }
        ? $"Accuracy (AI version): {af.EdgesKept * 100:0.0} % of lines and letters kept · {af.NewEdges * 100:0.0} % new marks"
        : Job.IsUnprovenRedraw
        ? "Accuracy: Check · redrawn but not verified: compare every atom, bond and stereo mark with the original"
        : Job.IsRedrawn
        ? "Accuracy: Verified by the automatic checks · the redraw reads back as the same molecule that was read from the picture " +
          "(same InChIKey, including stereochemistry). This is not proof that it matches the original: a chemist must compare it."
        : Job.Status != ProcessingStatus.Completed || Job.Fidelity is not { } f ? ""
        : $"Accuracy: {ReportBuilder.Rate(Job)} · {f.EdgesKept * 100:0.0} % of lines and letters kept · {f.NewEdges * 100:0.0} % new marks · {Wording.Level(Job.Settings.Strength)}";

    public bool HasAccuracy => AccuracyText.Length > 0;

    public string RedrawText => !ShowsChemistry ? ""
        : Job.Panels.Count > 0
        ? string.Join("\n", Job.Panels.Select(Wording.Panel))
        : Wording.Redraw(Job.Redraw, Job.ContentKind);

    // ------------------------------------------------------------------ category (chemical diagram, table, ...)

    /// <summary>
    /// Chemical details (structure checks, redraw outcome) belong to chemical pictures only. A picture that was
    /// actually redrawn always shows them, so a saved redraw is never left without its warning.
    /// </summary>
    public bool ShowsChemistry => Wording.ShowsChemistry(Job);

    /// <summary>"Table · 99 %", "Chemical diagram · set by you", ... for the list row.</summary>
    public string CategoryLine => Wording.CategoryLine(Job);

    /// <summary>Looks chemical but MolScribe could not read it reliably: kept with the chemical diagrams.</summary>
    public bool IsPossibleChemical => Job.IsPossibleChemical;

    public string PossibleChemicalText => "Possible chemical diagram — needs review";

    public string ClassificationDetails => Wording.ClassificationDetails(Job);

    /// <summary>Classified as chemical / read by MolScribe / possible chemical / verified structure; empty for other pictures.</summary>
    public string ChemicalStatusText => Wording.ChemicalStatus(Job);

    public bool HasChemicalStatus => ChemicalStatusText.Length > 0;

    /// <summary>The user changed the category: "Use the suggestion" brings the classifier's back.</summary>
    public bool HasManualCategory => Job.ManualCategory is not null;

    // ------------------------------------------------------------------ optional AI enhancement

    /// <summary>Before AI is tried: what may help (or why nothing is offered). After: the outcome.</summary>
    public string AiText => Job.Ai is null ? Wording.AiSuggestions(Job) : Wording.AiOutcome(Job);

    /// <summary>The colour that marks AI enhancement (badge, headline, picture label).</summary>
    public static readonly IBrush AiBrush = new SolidColorBrush(Color.Parse("#6B46C1"));

    /// <summary>"AI enhancement applied (SCUNet)", "... not applied", ...: shown once AI was tried.</summary>
    public string AiHeadline => Wording.AiHeadline(Job);

    public bool HasAiHeadline => Job.Ai is not null;

    /// <summary>✓ applied, i ready but not in use, ✕ not applied, ! could not run.</summary>
    public string AiHeadlineIcon => Job.Ai switch
    {
        { Accepted: true } when Job.UseAiVersion => "✓",
        { Accepted: true } => "i",
        { Status: "rejected" or "not_eligible" } => "✕",
        _ => "!",
    };

    public IBrush AiHeadlineBrush => Job.Ai switch
    {
        { Accepted: true } when Job.UseAiVersion => AiBrush,
        { Accepted: true } => Brushes.SteelBlue,
        { Status: "rejected" or "not_eligible" } => Brushes.DarkOrange,
        _ => Brushes.Firebrick,
    };

    public bool CanTryAi => JobOrchestrator.IsAiEligible(Job);

    /// <summary>An AI version exists (usable or not) and can be compared.</summary>
    public bool HasAiResult => Job.Ai?.Path is not null;

    public bool IsAiAccepted => Job.Ai is { Accepted: true };
    public bool UsesAi => Job.UseAiVersion;
    public bool UsesNormal => !Job.UseAiVersion;

    /// <summary>The normal version's name in the version choice: the OpenCV result, or the original when nothing needed fixing.</summary>
    public string NormalVersionLabel => Job.Status == ProcessingStatus.Completed ? "OpenCV result" : "Original (nothing to fix)";

    /// <summary>What AI did, in the list row: applied, ready but not in use, or tried and not applied.</summary>
    public string AiRowText => Job.Ai switch
    {
        { Accepted: true } ai when Job.UseAiVersion => $"✓ AI enhancement applied ({ai.ModelNames})",
        { Accepted: true } => "AI version ready, not in use",
        { Status: "rejected" } => "AI tried, not applied",
        _ => "",
    };


    public bool HasAiRowText => AiRowText.Length > 0;

    public string? AiImagePath => Abs(Job.Ai?.Path);
    public string? AiThumbnailPath => Abs(Job.Ai?.ThumbnailPath);

    /// <summary>Only doubtful pictures offer a category change; confident suggestions are simply used.</summary>
    public bool IsCategoryDoubtful => Wording.IsCategoryDoubtful(Job);

    public string CategoryQuestion => Wording.CategoryQuestion(Job);
    public bool HasRedraw => RedrawText.Length > 0;

    /// <summary>
    /// What did not match when a chemical drawing was checked for a redraw (not redrawn, or redrawn without full
    /// proof): a heading and a numbered list, numbered like the red boxes on the original picture.
    /// </summary>
    public string MismatchText
    {
        get
        {
            var items = Mismatches;
            if (items.Count == 0)
            {
                return "";
            }

            return string.Join(Environment.NewLine, items.Select(i =>
                $"{i.Number}. {i.Message}" + (i.Boxes.Count > 1 ? $" (marked {i.Boxes.Count}×)" : "")));
        }
    }

    public bool HasMismatches => MismatchText.Length > 0;

    /// <summary>"Structure: 3 mismatches" for the image list.</summary>
    public string MismatchSummary
    {
        get
        {
            var count = Mismatches.Count;
            return count == 0 ? "" : $"Structure checks · {count} item{(count == 1 ? "" : "s")} to review";
        }
    }

    public bool HasMismatchSummary => MismatchSummary.Length > 0;

    /// <summary>Numbered boxes on the original picture, where the mismatches are.</summary>
    /// <summary>Structure mismatches: chemical pictures only (tables, graphs and photos never show chemical errors).</summary>
    private IReadOnlyList<(int Number, string Message, IReadOnlyList<double[]> Boxes)> Mismatches =>
        ShowsChemistry ? Wording.Mismatches(Job) : [];

    public IReadOnlyList<ImageMark> MismatchMarks => Mismatches
        .SelectMany(i => i.Boxes.Select(b => new ImageMark(i.Number, new Avalonia.Rect(b[0], b[1], Math.Max(1, b[2] - b[0]), Math.Max(1, b[3] - b[1])))))
        .ToList();

    public bool HasMismatchMarks => MismatchMarks.Count > 0;

    public string TextCheckText => Job.Status == ProcessingStatus.Completed ? Wording.TextCheck(Job.TextCheck) : "";

    /// <summary>Which version was kept when AI repair made alternatives, and where the AI ran.</summary>
    public string ChoiceText => Job.Status != ProcessingStatus.Completed ? ""
        : string.Join("\n", new[] { Wording.Choice(Job.Choice), Wording.Devices(Job.AiDevices) }.Where(t => t.Length > 0));

    public bool HasChoice => ChoiceText.Length > 0;
    public bool HasTextCheck => TextCheckText.Length > 0;

    // The level this result was made with (automatic mode), for the level buttons.
    private bool AtLevel(string key) => Job.Settings.Mode == "auto" && Job.Settings.Strength == key
                                        && Job.Status is ProcessingStatus.Completed or ProcessingStatus.Skipped;
    public bool IsLevel1 => AtLevel("gentle");
    public bool IsLevel2 => AtLevel("standard");
    public bool IsLevel3 => AtLevel("strong");
    public bool IsLevel4 => AtLevel("maximum");

    public string NotesText => string.Join("; ", Job.Notes.Select(n => Capitalise(n)));

    // ------------------------------------------------------------------ images

    public string? OriginalThumbnailPath => Abs(Job.OriginalThumbnailPath);
    public string? EnhancedThumbnailPath => Abs(Job.EnhancedThumbnailPath);

    /// <summary>The original pixels: the extracted copy for documents, the user's file for images (read only).</summary>
    public string? OriginalImagePath => Job.ExtractedPath is not null ? Abs(Job.ExtractedPath) : Job.Location.SourcePath;

    public string? EnhancedImagePath => Abs(Job.EnhancedPath);

    public Bitmap? OriginalThumbnail => ImageCache.Thumbnails.Get(OriginalThumbnailPath, () => Changed(nameof(OriginalThumbnail)));
    public Bitmap? EnhancedThumbnail => ImageCache.Thumbnails.Get(EnhancedThumbnailPath, () => Changed(nameof(EnhancedThumbnail)));

    /// <summary>The gallery row's picture: the enhanced result when there is one, otherwise the original.</summary>
    public Bitmap? Thumbnail => Job.UseAiVersion && AiThumbnailPath is not null
        ? ImageCache.Thumbnails.Get(AiThumbnailPath, () => Changed(nameof(Thumbnail)))
        : EnhancedThumbnailPath is not null
        ? ImageCache.Thumbnails.Get(EnhancedThumbnailPath, () => Changed(nameof(Thumbnail)))
        : ImageCache.Thumbnails.Get(OriginalThumbnailPath, () => Changed(nameof(Thumbnail)));

    private void Changed(string property) => Dispatcher.UIThread.Post(() => OnPropertyChanged(property));

    private string? Abs(string? relative) => relative is null ? null : Path.Combine(_workspaceRoot, relative);

    private static string DescribeReview(string reason) => Wording.ReviewReason(reason);

    private static string? Capitalise(string? text) =>
        string.IsNullOrEmpty(text) ? text : char.ToUpperInvariant(text[0]) + text[1..];
}
