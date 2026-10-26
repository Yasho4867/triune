using Microsoft.Web.WebView2.Core;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Windows;

namespace TriuneStudio.Windows;

public partial class MainWindow : Window
{
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private string? _workspaceRoot;
    private string? _pythonPath;
    private Process? _backendProcess;
    private StreamWriter? _backendLog;
    private string? _logPath;
    private int _port = 8000;
    private bool _closing;
    private bool _backendRestarting;
    private Task? _backendMonitor;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += async (_, _) => await StartStudioAsync();
    }

    private async Task StartStudioAsync()
    {
        RetryButton.Visibility = Visibility.Collapsed;
        SetupButton.Visibility = Visibility.Collapsed;
        OpenLogsButton.Visibility = Visibility.Collapsed;
        StartupProgress.Visibility = Visibility.Visible;
        StatusText.Text = "Starting the local model engine…";

        try
        {
            _workspaceRoot ??= FindWorkspaceRoot();
            _pythonPath ??= FindPython(_workspaceRoot);
            _port = await FindStudioPortAsync();
            StartBackend(_workspaceRoot, _pythonPath, _port);

            var baseUri = new Uri($"http://127.0.0.1:{_port}/");
            // Match the launcher's 90-second cold-start allowance for WSL and
            // leave room for its final health check before reporting failure.
            if (!await WaitForBackendReadyAsync(baseUri, TimeSpan.FromSeconds(105)))
                throw new TimeoutException("The backend did not become ready within 105 seconds.");

            await ShowStudioAsync(baseUri);
            _backendMonitor ??= MonitorBackendAsync();
        }
        catch (Exception ex)
        {
            // A failed startup must not leave an orphaned process holding the
            // port. Otherwise every retry starts another backend on a new port.
            var hasLog = _logPath is not null;
            await StopBackendAsync();
            StartupProgress.Visibility = Visibility.Collapsed;
            RetryButton.Visibility = Visibility.Visible;
            SetupButton.Visibility = Visibility.Visible;
            OpenLogsButton.Visibility = hasLog ? Visibility.Visible : Visibility.Collapsed;
            StatusText.Text = $"Studio could not start.\n\n{ex.Message}\n\nUse Install / repair to provision the managed Python, PyTorch/CUDA, and Studio environment, then retry.";
        }
    }

    private async Task<bool> WaitForBackendReadyAsync(Uri baseUri, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        var healthUri = new Uri(baseUri, "healthz");
        while (!_closing && DateTime.UtcNow < deadline)
        {
            if (_backendProcess is { HasExited: true }) return false;
            try
            {
                using var response = await _http.GetAsync(healthUri);
                if (response.IsSuccessStatusCode) return true;
            }
            catch (HttpRequestException) { }
            catch (TaskCanceledException) { }
            await Task.Delay(350);
        }
        return false;
    }

    private async Task MonitorBackendAsync()
    {
        var failures = 0;
        while (!_closing)
        {
            await Task.Delay(TimeSpan.FromSeconds(2));
            if (_closing) return;

            var healthy = false;
            if (_backendProcess is { HasExited: false })
            {
                try
                {
                    using var response = await _http.GetAsync(new Uri($"http://127.0.0.1:{_port}/healthz"));
                    healthy = response.IsSuccessStatusCode;
                }
                catch (HttpRequestException) { }
                catch (TaskCanceledException) { }
            }

            failures = healthy ? 0 : failures + 1;
            if (failures >= 4)
            {
                await RecoverBackendAsync();
                failures = 0;
            }
        }
    }

    private async Task RecoverBackendAsync()
    {
        if (_backendRestarting || _closing || _workspaceRoot is null || _pythonPath is null) return;
        _backendRestarting = true;
        try
        {
            for (var attempt = 1; !_closing; attempt++)
            {
                await StopBackendAsync();
                try
                {
                    _port = await FindStudioPortAsync();
                    StartBackend(_workspaceRoot, _pythonPath, _port);
                    var baseUri = new Uri($"http://127.0.0.1:{_port}/");
                    if (await WaitForBackendReadyAsync(baseUri, TimeSpan.FromSeconds(105)))
                    {
                        StudioView.Source = baseUri;
                        return;
                    }
                    AppendBackendLog("Backend restart did not become healthy within 105 seconds.");
                }
                catch (Exception ex)
                {
                    AppendBackendLog($"Backend restart attempt failed: {ex.Message}");
                }

                await StopBackendAsync();
                var backoffSeconds = Math.Min(30, 1 << Math.Min(attempt, 5));
                await Task.Delay(TimeSpan.FromSeconds(backoffSeconds));
            }
        }
        finally
        {
            _backendRestarting = false;
        }
    }

    private void AppendBackendLog(string message)
    {
        if (_logPath is null) return;
        try { File.AppendAllText(_logPath, $"{DateTimeOffset.Now:O} {message}{Environment.NewLine}"); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private async Task ShowStudioAsync(Uri baseUri)
    {
        StatusText.Text = "Opening Studio…";
        await StudioView.EnsureCoreWebView2Async();
        try
        {
            await StudioView.CoreWebView2.Profile.ClearBrowsingDataAsync(
                CoreWebView2BrowsingDataKinds.DiskCache | CoreWebView2BrowsingDataKinds.CacheStorage);
        }
        catch (Exception) { }
        StudioView.CoreWebView2.Settings.AreDefaultContextMenusEnabled = true;
        StudioView.CoreWebView2.Settings.IsStatusBarEnabled = false;
        StudioView.CoreWebView2.Settings.IsZoomControlEnabled = false;
        StudioView.CoreWebView2.Settings.IsPinchZoomEnabled = false;
        StudioView.CoreWebView2.NavigationStarting += (_, args) =>
        {
            if (Uri.TryCreate(args.Uri, UriKind.Absolute, out var target) &&
                IPAddress.TryParse(target.Host, out var address) && IPAddress.IsLoopback(address) && target.Port == _port)
                return;

            args.Cancel = true;
            OpenExternal(args.Uri);
        };
        StudioView.CoreWebView2.NewWindowRequested += (_, args) =>
        {
            args.Handled = true;
            OpenExternal(args.Uri);
        };

        StartupPanel.Visibility = Visibility.Collapsed;
        StudioView.Visibility = Visibility.Visible;
        StudioView.Source = baseUri;
    }

    private static string FindWorkspaceRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "pyproject.toml")) &&
                File.Exists(Path.Combine(directory.FullName, "studio", "src", "index.html")))
                return directory.FullName;
        }

        throw new DirectoryNotFoundException("Could not locate TriuneTransformer. Build or run this app from the repository installation.");
    }

    private static string FindPython(string root)
    {
        var candidates = new[]
        {
            Path.Combine(root, "studio", "studio_env", "Scripts", "python.exe"),
            Path.Combine(root, ".venv", "Scripts", "python.exe")
        };
        foreach (var candidate in candidates)
            if (File.Exists(candidate)) return candidate;

        foreach (var command in new[] { "python.exe", "python", "py.exe", "py" })
        {
            var path = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator)
                .Select(folder => Path.Combine(folder.Trim('"'), command))
                .FirstOrDefault(File.Exists);
            if (path is not null) return path;
        }

        throw new FileNotFoundException("Python was not found. Install Studio from the project environment first.");
    }

    private Task<int> FindStudioPortAsync()
    {
        for (var port = 8000; port <= 8019; port++)
        {
            // A responsive API may be an orphaned backend from an earlier
            // launch (possibly running without this app's credential bridge).
            // Native Studio owns its backend lifecycle, so always prefer a free
            // port and start a fresh process instead of silently attaching.
            var listener = new TcpListener(IPAddress.Loopback, port);
            try
            {
                listener.Start();
                return Task.FromResult(port);
            }
            catch (SocketException) { }
            finally { listener.Stop(); }
        }

        throw new IOException("No free local Studio port is available (8000–8019).");
    }

    private void StartBackend(string root, string python, int port)
    {
        var logsDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Triune", "logs");
        Directory.CreateDirectory(logsDirectory);
        _logPath = Path.Combine(logsDirectory, $"studio-native-{DateTime.Now:yyyyMMdd-HHmmss}.log");

        var start = new ProcessStartInfo
        {
            FileName = python,
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        if (Path.GetFileNameWithoutExtension(python).Equals("py", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add("-3");
        start.ArgumentList.Add(Path.Combine(root, "scripts", "launch_studio.py"));
        start.ArgumentList.Add("--port");
        start.ArgumentList.Add(port.ToString());
        start.ArgumentList.Add("--native-host");

        _backendLog = new StreamWriter(_logPath, append: true) { AutoFlush = true };
        _backendProcess = new Process { StartInfo = start, EnableRaisingEvents = true };
        _backendProcess.OutputDataReceived += (_, e) => { if (e.Data is not null && _backendLog is { } log) lock (log) log.WriteLine(e.Data); };
        _backendProcess.ErrorDataReceived += (_, e) => { if (e.Data is not null && _backendLog is { } log) lock (log) log.WriteLine(e.Data); };
        if (!_backendProcess.Start())
            throw new InvalidOperationException("Could not start the Python backend process.");
        _backendProcess.BeginOutputReadLine();
        _backendProcess.BeginErrorReadLine();
    }

    private static void OpenExternal(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            (uri.Scheme != Uri.UriSchemeHttps && uri.Scheme != Uri.UriSchemeHttp)) return;
        Process.Start(new ProcessStartInfo(uri.ToString()) { UseShellExecute = true });
    }

    private async void RetryButton_Click(object sender, RoutedEventArgs e)
    {
        await StopBackendAsync();
        await StartStudioAsync();
    }

    private async void SetupButton_Click(object sender, RoutedEventArgs e)
    {
        if (_workspaceRoot is null) return;
        var installer = Path.Combine(_workspaceRoot, "studio", "installer", "install_studio.bat");
        if (!File.Exists(installer))
        {
            StatusText.Text = "The Studio setup script is missing from this installation.";
            return;
        }

        SetupButton.IsEnabled = false;
        RetryButton.IsEnabled = false;
        OpenLogsButton.Visibility = Visibility.Collapsed;
        StartupProgress.Visibility = Visibility.Visible;
        StatusText.Text = "Installing or repairing Studio. This can take a while while PyTorch and GPU support are prepared…";
        string? setupLog = null;
        try
        {
            var logsDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Triune", "logs");
            Directory.CreateDirectory(logsDirectory);
            var setupLogPath = Path.Combine(logsDirectory, $"studio-setup-{DateTime.Now:yyyyMMdd-HHmmss}.log");
            setupLog = setupLogPath;
            var setupLogLock = new object();
            var start = new ProcessStartInfo("cmd.exe")
            {
                WorkingDirectory = _workspaceRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            start.ArgumentList.Add("/c");
            start.ArgumentList.Add($"\"{installer}\" --yes --no-launch");
            using var setup = new Process { StartInfo = start };
            setup.OutputDataReceived += (_, args) => { if (args.Data is not null) lock (setupLogLock) File.AppendAllText(setupLogPath, args.Data + Environment.NewLine); };
            setup.ErrorDataReceived += (_, args) => { if (args.Data is not null) lock (setupLogLock) File.AppendAllText(setupLogPath, args.Data + Environment.NewLine); };
            if (!setup.Start()) throw new InvalidOperationException("Could not start the setup process.");
            setup.BeginOutputReadLine();
            setup.BeginErrorReadLine();
            await setup.WaitForExitAsync();
            if (setup.ExitCode != 0)
                throw new InvalidOperationException($"Setup exited with code {setup.ExitCode}. Open the setup log: {setupLog}");

            _workspaceRoot = null;
            _pythonPath = null;
            await StartStudioAsync();
        }
        catch (Exception ex)
        {
            StartupProgress.Visibility = Visibility.Collapsed;
            StatusText.Text = $"Studio setup failed.\n\n{ex.Message}" + (setupLog is null ? "" : $"\n\nSetup log: {setupLog}");
            OpenLogsButton.Visibility = setupLog is null ? Visibility.Collapsed : Visibility.Visible;
            _logPath = setupLog;
        }
        finally
        {
            SetupButton.IsEnabled = true;
            RetryButton.IsEnabled = true;
        }
    }

    private void OpenLogsButton_Click(object sender, RoutedEventArgs e)
    {
        if (_logPath is not null && File.Exists(_logPath))
            Process.Start(new ProcessStartInfo("notepad.exe", $"\"{_logPath}\"") { UseShellExecute = true });
    }

    private async void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_closing) return;
        _closing = true;
        await StopBackendAsync();
        _http.Dispose();
    }

    private async Task StopBackendAsync()
    {
        if (_backendProcess is null) return;
        try
        {
            if (!_backendProcess.HasExited)
            {
                _backendProcess.Kill(entireProcessTree: true);
                await _backendProcess.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5));
            }
        }
        catch (InvalidOperationException) { }
        catch (TimeoutException) { }
        catch (System.ComponentModel.Win32Exception) { }
        _backendProcess.Dispose();
        _backendProcess = null;
        _backendLog?.Dispose();
        _backendLog = null;
    }
}
