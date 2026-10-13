using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using ImageQualityEnhancer.ViewModels;

namespace ImageQualityEnhancer.Views;

public partial class JobView : UserControl
{
    public JobView()
    {
        InitializeComponent();
    }

    private JobViewModel? ViewModel => DataContext as JobViewModel;

    /// <summary>The viewer currently on screen; all viewers share zoom and position through the view model.</summary>
    private ZoomCompareView? ActiveViewer => ViewModel?.IsSlider == true ? SliderView : EnhancedView;

    private void ZoomBy(double factor)
    {
        if (ActiveViewer is { } viewer)
        {
            viewer.ZoomAt(factor, new Point(viewer.Bounds.Width / 2, viewer.Bounds.Height / 2));
        }
    }

    private void OnZoomIn(object? sender, RoutedEventArgs e) => ZoomBy(1.25);

    private void OnZoomOut(object? sender, RoutedEventArgs e) => ZoomBy(0.8);

    /// <summary>Keyboard shortcuts for reviewing quickly. Ignored while typing in a text box.</summary>
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (e.Handled || ViewModel is not { } vm || TopLevel.GetTopLevel(this)?.FocusManager?.GetFocusedElement() is TextBox)
        {
            return;
        }

        switch (e.Key)
        {
            case Key.Add or Key.OemPlus:
                ZoomBy(1.25);
                break;
            case Key.Subtract or Key.OemMinus:
                ZoomBy(0.8);
                break;
            case Key.D0 or Key.NumPad0:
                vm.ZoomFitCommand.Execute(null);
                break;
            case Key.D1 or Key.NumPad1:
                ActiveViewer?.ZoomTo(1.0);
                break;
            case Key.A when vm.SelectedItem is { CanApprove: true }:
                vm.ApproveCommand.Execute(vm.SelectedItem);
                break;
            case Key.K when vm.SelectedItem is { CanApprove: true }:
                vm.KeepOriginalCommand.Execute(vm.SelectedItem);
                break;
            case Key.E:
                vm.ToggleExpandedCommand.Execute(null);
                break;
            case Key.S:
                vm.SetViewModeCommand.Execute(vm.IsSlider ? "side" : "slider");
                break;
            case Key.Up:
                vm.SelectNextCommand.Execute("previous");
                break;
            case Key.Down:
                vm.SelectNextCommand.Execute("next");
                break;
            default:
                return;
        }

        e.Handled = true;
    }
}
