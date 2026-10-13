using System;
using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace ImageQualityEnhancer.Views;

/// <summary>
/// Zoomable, pannable image view for inspecting results. Mouse wheel zooms around the cursor, dragging pans,
/// double-click toggles between fit and 100 %. With <see cref="CompareSource"/> set and <see cref="SplitMode"/> on,
/// it shows a before/after slider: the compare image left of a draggable divider, the source to its right.
///
/// <see cref="Zoom"/> (0 = fit) and <see cref="Center"/> (normalised image point at the view centre) bind two-way,
/// so several views can share them and stay in sync. Images of different pixel sizes are drawn over the same area,
/// so an enhanced image with more pixels lines up with its original.
/// </summary>
public sealed class ZoomCompareView : Control
{
    public static readonly StyledProperty<Bitmap?> SourceProperty =
        AvaloniaProperty.Register<ZoomCompareView, Bitmap?>(nameof(Source));

    public static readonly StyledProperty<Bitmap?> CompareSourceProperty =
        AvaloniaProperty.Register<ZoomCompareView, Bitmap?>(nameof(CompareSource));

    public static readonly StyledProperty<bool> SplitModeProperty =
        AvaloniaProperty.Register<ZoomCompareView, bool>(nameof(SplitMode));

    public static readonly StyledProperty<double> SplitPositionProperty =
        AvaloniaProperty.Register<ZoomCompareView, double>(nameof(SplitPosition), 0.5, defaultBindingMode: BindingMode.TwoWay);

    /// <summary>Device-independent pixels per image pixel of the reference image; 0 means "fit to view".</summary>
    public static readonly StyledProperty<double> ZoomProperty =
        AvaloniaProperty.Register<ZoomCompareView, double>(nameof(Zoom), 0, defaultBindingMode: BindingMode.TwoWay);

    public static readonly StyledProperty<Point> CenterProperty =
        AvaloniaProperty.Register<ZoomCompareView, Point>(nameof(Center), new Point(0.5, 0.5), defaultBindingMode: BindingMode.TwoWay);

    /// <summary>Pixel size both images are drawn at (the larger, enhanced image); falls back to the source's size.</summary>
    public static readonly StyledProperty<Size> ReferenceSizeProperty =
        AvaloniaProperty.Register<ZoomCompareView, Size>(nameof(ReferenceSize));

    /// <summary>Numbered boxes drawn over the image, in the reference image's pixels (for example mismatches).</summary>
    public static readonly StyledProperty<IReadOnlyList<ImageQualityEnhancer.ViewModels.ImageMark>?> MarksProperty =
        AvaloniaProperty.Register<ZoomCompareView, IReadOnlyList<ImageQualityEnhancer.ViewModels.ImageMark>?>(nameof(Marks));

    public static readonly StyledProperty<IBrush?> BackgroundProperty =
        AvaloniaProperty.Register<ZoomCompareView, IBrush?>(nameof(Background));

    public const double MaxZoom = 16;

    private Point? _dragStart;
    private Point _centerAtDragStart;
    private bool _draggingSplit;

    static ZoomCompareView()
    {
        AffectsRender<ZoomCompareView>(SourceProperty, CompareSourceProperty, SplitModeProperty, SplitPositionProperty,
            ZoomProperty, CenterProperty, ReferenceSizeProperty, BackgroundProperty, MarksProperty);
        FocusableProperty.OverrideDefaultValue<ZoomCompareView>(true);
        ClipToBoundsProperty.OverrideDefaultValue<ZoomCompareView>(true);
    }

    public Bitmap? Source { get => GetValue(SourceProperty); set => SetValue(SourceProperty, value); }
    public Bitmap? CompareSource { get => GetValue(CompareSourceProperty); set => SetValue(CompareSourceProperty, value); }
    public bool SplitMode { get => GetValue(SplitModeProperty); set => SetValue(SplitModeProperty, value); }
    public double SplitPosition { get => GetValue(SplitPositionProperty); set => SetValue(SplitPositionProperty, value); }
    public double Zoom { get => GetValue(ZoomProperty); set => SetValue(ZoomProperty, value); }
    public Point Center { get => GetValue(CenterProperty); set => SetValue(CenterProperty, value); }
    public Size ReferenceSize { get => GetValue(ReferenceSizeProperty); set => SetValue(ReferenceSizeProperty, value); }
    public IBrush? Background { get => GetValue(BackgroundProperty); set => SetValue(BackgroundProperty, value); }
    public IReadOnlyList<ImageQualityEnhancer.ViewModels.ImageMark>? Marks { get => GetValue(MarksProperty); set => SetValue(MarksProperty, value); }

    private Size Logical
    {
        get
        {
            if (ReferenceSize.Width > 0 && ReferenceSize.Height > 0)
            {
                return ReferenceSize;
            }

            var bitmap = Source ?? CompareSource;
            return bitmap is null ? default : new Size(bitmap.PixelSize.Width, bitmap.PixelSize.Height);
        }
    }

    private double FitScale
    {
        get
        {
            var logical = Logical;
            return logical.Width <= 0 || Bounds.Width <= 0
                ? 1
                : Math.Min(Bounds.Width / logical.Width, Bounds.Height / logical.Height);
        }
    }

    private double Scale => Zoom <= 0 ? FitScale : Zoom;

    /// <summary>Where the whole image is drawn, in view coordinates.</summary>
    private Rect ImageRect
    {
        get
        {
            var logical = Logical;
            var scale = Scale;
            var width = logical.Width * scale;
            var height = logical.Height * scale;
            var center = Zoom <= 0 ? new Point(0.5, 0.5) : Center;
            return new Rect(Bounds.Width / 2 - center.X * width, Bounds.Height / 2 - center.Y * height, width, height);
        }
    }

    private double SplitX => ImageRect.X + ImageRect.Width * Math.Clamp(SplitPosition, 0, 1);

    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        if (Background is not null)
        {
            context.FillRectangle(Background, bounds);
        }

        if (Logical.Width <= 0)
        {
            return;
        }

        var dest = ImageRect;
        // Show real pixels when zoomed in far, smooth scaling otherwise.
        var mode = Scale >= 2.5 ? BitmapInterpolationMode.None : BitmapInterpolationMode.HighQuality;
        using (context.PushRenderOptions(new RenderOptions { BitmapInterpolationMode = mode }))
        {
            if (SplitMode && CompareSource is not null && Source is not null)
            {
                var split = SplitX;
                using (context.PushClip(new Rect(0, 0, Math.Max(0, split), bounds.Height)))
                {
                    Draw(context, CompareSource, dest);
                }

                using (context.PushClip(new Rect(split, 0, Math.Max(0, bounds.Width - split), bounds.Height)))
                {
                    Draw(context, Source, dest);
                }

                DrawDivider(context, split, bounds.Height);
            }
            else if (Source is not null)
            {
                Draw(context, Source, dest);
            }
        }

        DrawMarks(context, dest);
    }

    private static readonly IBrush MarkBrush = new SolidColorBrush(Color.FromRgb(220, 38, 38));
    private static readonly IPen MarkPen = new Pen(MarkBrush, 2.5);

    /// <summary>Red boxes with their number at the top-left corner, scaled and moved with the image.</summary>
    private void DrawMarks(DrawingContext context, Rect dest)
    {
        if (Marks is not { Count: > 0 } marks || Logical.Width <= 0)
        {
            return;
        }

        var sx = dest.Width / Logical.Width;
        var sy = dest.Height / Logical.Height;
        foreach (var mark in marks)
        {
            var box = new Rect(dest.X + mark.Box.X * sx, dest.Y + mark.Box.Y * sy, mark.Box.Width * sx, mark.Box.Height * sy);
            context.DrawRectangle(null, MarkPen, box, 3, 3);
            var text = new FormattedText(mark.Number.ToString(System.Globalization.CultureInfo.InvariantCulture),
                System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, 12, Brushes.White);
            var badge = new Rect(box.X - 1, box.Y - text.Height - 2, text.Width + 8, text.Height + 2);
            context.DrawRectangle(MarkBrush, null, badge, 3, 3);
            context.DrawText(text, new Point(badge.X + 4, badge.Y + 1));
        }
    }

    private static void Draw(DrawingContext context, Bitmap bitmap, Rect dest) =>
        context.DrawImage(bitmap, new Rect(0, 0, bitmap.PixelSize.Width, bitmap.PixelSize.Height), dest);

    private static readonly IPen DividerPen = new Pen(Brushes.White, 2);
    private static readonly IPen DividerShadow = new Pen(new SolidColorBrush(Color.FromArgb(120, 0, 0, 0)), 4);

    private static void DrawDivider(DrawingContext context, double x, double height)
    {
        context.DrawLine(DividerShadow, new Point(x, 0), new Point(x, height));
        context.DrawLine(DividerPen, new Point(x, 0), new Point(x, height));
        var knob = new Point(x, height / 2);
        context.DrawEllipse(Brushes.White, DividerShadow, knob, 12, 12);
        var arrow = new Pen(Brushes.Black, 1.5);
        context.DrawLine(arrow, new Point(x - 6, knob.Y), new Point(x - 2, knob.Y - 4));
        context.DrawLine(arrow, new Point(x - 6, knob.Y), new Point(x - 2, knob.Y + 4));
        context.DrawLine(arrow, new Point(x + 6, knob.Y), new Point(x + 2, knob.Y - 4));
        context.DrawLine(arrow, new Point(x + 6, knob.Y), new Point(x + 2, knob.Y + 4));
    }

    // ------------------------------------------------------------------ interaction

    /// <summary>Zooms by <paramref name="factor"/>, keeping the image point under <paramref name="anchor"/> still.</summary>
    public void ZoomAt(double factor, Point anchor)
    {
        var rect = ImageRect;
        if (rect.Width <= 0)
        {
            return;
        }

        var imagePoint = new Point((anchor.X - rect.X) / rect.Width, (anchor.Y - rect.Y) / rect.Height);
        var newScale = Math.Clamp(Scale * factor, FitScale, Math.Max(FitScale, MaxZoom));
        if (Math.Abs(newScale - FitScale) < 1e-6)
        {
            Zoom = 0;
            Center = new Point(0.5, 0.5);
            return;
        }

        var logical = Logical;
        var width = logical.Width * newScale;
        var height = logical.Height * newScale;
        Zoom = newScale;
        Center = Clamp(new Point(
            imagePoint.X - (anchor.X - Bounds.Width / 2) / width,
            imagePoint.Y - (anchor.Y - Bounds.Height / 2) / height));
    }

    /// <summary>Sets an absolute zoom (1.0 = actual pixels of the reference image), keeping the current centre.</summary>
    public void ZoomTo(double scale)
    {
        if (Bounds.Width <= 0)
        {
            return;
        }

        ZoomAt(scale / Scale, new Point(Bounds.Width / 2, Bounds.Height / 2));
    }

    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        ZoomAt(e.Delta.Y > 0 ? 1.25 : 0.8, e.GetPosition(this));
        e.Handled = true;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
        {
            return;
        }

        Focus();
        var position = e.GetPosition(this);
        if (e.ClickCount == 2)
        {
            if (Zoom <= 0)
            {
                ZoomAt(1.0 / FitScale, position); // 100 % at the clicked point
            }
            else
            {
                Zoom = 0;
                Center = new Point(0.5, 0.5);
            }

            e.Handled = true;
            return;
        }

        _draggingSplit = SplitMode && CompareSource is not null && Math.Abs(position.X - SplitX) <= 14;
        _dragStart = position;
        _centerAtDragStart = Zoom <= 0 ? new Point(0.5, 0.5) : Center;
        e.Pointer.Capture(this);
        e.Handled = true;
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var position = e.GetPosition(this);
        var nearSplit = SplitMode && CompareSource is not null && Math.Abs(position.X - SplitX) <= 14;
        Cursor = new Cursor(nearSplit || _draggingSplit ? StandardCursorType.SizeWestEast
            : Zoom > 0 ? StandardCursorType.Hand : StandardCursorType.Arrow);

        if (_dragStart is not { } start)
        {
            return;
        }

        var rect = ImageRect;
        if (_draggingSplit)
        {
            SplitPosition = Math.Clamp((position.X - rect.X) / Math.Max(rect.Width, 1), 0, 1);
            return;
        }

        if (Zoom <= 0 || rect.Width <= 0)
        {
            return; // nothing to pan when the whole image is visible
        }

        Center = Clamp(new Point(
            _centerAtDragStart.X - (position.X - start.X) / rect.Width,
            _centerAtDragStart.Y - (position.Y - start.Y) / rect.Height));
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _dragStart = null;
        _draggingSplit = false;
        e.Pointer.Capture(null);
    }

    private static Point Clamp(Point p) => new(Math.Clamp(p.X, 0, 1), Math.Clamp(p.Y, 0, 1));
}
