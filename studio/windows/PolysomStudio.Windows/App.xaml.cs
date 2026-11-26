using System.Runtime.InteropServices;
using System.Windows;

namespace PolysomStudio.Windows;

public partial class App : Application
{
    private const int AttachParentProcess = -1;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AttachConsole(int dwProcessId);

    private void Application_Startup(object sender, StartupEventArgs e)
    {
        if (e.Args.Any(a => a.Equals("--smoke-test", StringComparison.OrdinalIgnoreCase)))
        {
            ShutdownMode = ShutdownMode.OnExplicitShutdown;
            AttachConsole(AttachParentProcess);
            var exitCode = RunSmokeTest();
            Shutdown(exitCode);
            return;
        }

        var window = new MainWindow();
        window.Show();
    }

    private static int RunSmokeTest()
    {
        try
        {
            var iconInfo = GetResourceStream(new Uri("pack://application:,,,/polysom.ico"));
            if (iconInfo?.Stream is null || iconInfo.Stream.Length == 0)
            {
                Console.Error.WriteLine("[SMOKE FAIL] Embedded polysom.ico resource is missing or empty.");
                return 1;
            }
            iconInfo.Stream.Dispose();

            var workspaceRoot = PolysomStudio.Windows.MainWindow.FindWorkspaceRoot();
            var engineStatus = PolysomStudio.Windows.MainWindow.NativeCoreInterop.TryGetEngineStatus(workspaceRoot);
            if (string.IsNullOrWhiteSpace(engineStatus))
            {
                Console.Error.WriteLine($"[SMOKE FAIL] NativeCoreInterop.TryGetEngineStatus returned null for workspace '{workspaceRoot}'.");
                return 1;
            }

            Console.WriteLine($"[SMOKE OK] PolysomStudio.Windows | workspace={workspaceRoot} | engine={engineStatus} | icon=embedded");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[SMOKE FAIL] {ex.Message}");
            return 1;
        }
    }
}
