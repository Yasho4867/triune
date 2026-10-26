using System.Windows;

namespace TriuneStudio.Installer;

public partial class App : Application
{
    public static string[] StartupArgs { get; private set; } = [];

    private void Application_Startup(object sender, StartupEventArgs e)
    {
        StartupArgs = e.Args;
        var window = new MainWindow();
        window.Show();
    }
}
