using System.Runtime.InteropServices;
using System.Windows;

namespace PolysomStudio.Installer;

public partial class App : Application
{
    private const int AttachParentProcess = -1;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AttachConsole(int dwProcessId);

    public static string[] StartupArgs { get; private set; } = [];

    private async void Application_Startup(object sender, StartupEventArgs e)
    {
        StartupArgs = e.Args;

        var smokeTest = HasArg(e.Args, "--smoke-test");
        var verifyOnly = HasArg(e.Args, "--verify");
        var uninstall = HasArg(e.Args, "--uninstall");
        var yes = HasArg(e.Args, "--yes") || HasArg(e.Args, "-y");
        var noLaunch = HasArg(e.Args, "--no-launch");
        var headlessRequested = smokeTest || verifyOnly || (uninstall && yes) || (yes && noLaunch);

        if (headlessRequested)
        {
            ShutdownMode = ShutdownMode.OnExplicitShutdown;
            AttachConsole(AttachParentProcess);

            var exitCode = 1;
            try
            {
                exitCode = await PolysomStudio.Installer.MainWindow.RunHeadlessCliAsync(e.Args);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] {ex.Message}");
                exitCode = 1;
            }

            Shutdown(exitCode);
            return;
        }

        var window = new MainWindow();
        window.Show();
    }

    internal static bool HasArg(string[] args, string flag) =>
        args.Any(a => a.Equals(flag, StringComparison.OrdinalIgnoreCase));
}
