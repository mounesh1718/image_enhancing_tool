namespace ImageQualityEnhancer.Tests;

/// <summary>A unique directory that is removed when the test finishes.</summary>
internal sealed class TempDirectory : IDisposable
{
    public TempDirectory()
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "iqe-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public void Dispose()
    {
        // Windows briefly locks files that were just written or read (virus scanner, a preview still loading in the
        // background): try again for up to 3 seconds. A file that stays open still fails the test.
        for (var attempt = 0; Directory.Exists(Path); attempt++)
        {
            try
            {
                Directory.Delete(Path, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException && attempt < 30)
            {
                Thread.Sleep(100);
            }
        }
    }
}
