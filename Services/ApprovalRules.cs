using System.Collections.Generic;
using System.Linq;
using ImageQualityEnhancer.Models;

namespace ImageQualityEnhancer.Services;

/// <summary>Who may be approved, and the bulk approval actions offered in the gallery.</summary>
public static class ApprovalRules
{
    /// <summary>Only a finished enhancement can be approved.</summary>
    public static bool CanApprove(ImageJob image) =>
        image.ResultPath is not null && (image.Status == ProcessingStatus.Completed || image.UseAiVersion);

    /// <summary>
    /// Eligible for "Approve all": finished and not flagged. Text, diagrams, chemical structures, large changes and
    /// possible artefacts are flagged and always need an individual decision.
    /// </summary>
    public static bool IsEligibleForBulkApproval(ImageJob image) => CanApprove(image) && !image.NeedsReview
        // A chemical redraw (verified or not) is never approved automatically: a chemist approves each one.
        && !image.Operations.Contains("redraw") && !image.Panels.Any(p => p.Redraw is { Redrawn: true })
        // Neither is a picture that looks chemical but could not be read reliably, nor an AI version.
        && !image.IsPossibleChemical && !image.UseAiVersion;

    /// <summary>Returns false (and changes nothing) when the image cannot be approved.</summary>
    public static bool Set(ImageJob image, ApprovalStatus approval)
    {
        if (approval == ApprovalStatus.Approved && !CanApprove(image))
        {
            return false;
        }

        image.Approval = approval;
        return true;
    }

    /// <summary>Approves every eligible, still-undecided image. Earlier individual decisions are kept.</summary>
    public static int ApproveAllEligible(IEnumerable<ImageJob> images)
    {
        var count = 0;
        foreach (var image in images.Where(i => i.Approval == ApprovalStatus.Undecided && IsEligibleForBulkApproval(i)))
        {
            image.Approval = ApprovalStatus.Approved;
            count++;
        }

        return count;
    }

    /// <summary>Keeps the original for every image.</summary>
    public static int RejectAll(IEnumerable<ImageJob> images)
    {
        var count = 0;
        foreach (var image in images.Where(i => i.Approval != ApprovalStatus.KeepOriginal))
        {
            image.Approval = ApprovalStatus.KeepOriginal;
            count++;
        }

        return count;
    }

    /// <summary>Images that will be written into the output document.</summary>
    public static IEnumerable<ImageJob> ToReplace(IEnumerable<ImageJob> images) =>
        images.Where(i => i.Approval == ApprovalStatus.Approved && CanApprove(i));
}
