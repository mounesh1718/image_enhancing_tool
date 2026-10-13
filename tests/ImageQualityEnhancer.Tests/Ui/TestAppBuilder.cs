using Avalonia;
using Avalonia.Headless;
using ImageQualityEnhancer.Tests.Ui;

[assembly: AvaloniaTestApplication(typeof(TestAppBuilder))]

namespace ImageQualityEnhancer.Tests.Ui;

/// <summary>Runs the real App off-screen with Skia rendering, so UI tests can also capture screenshots.</summary>
public static class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UseSkia()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false });
}
