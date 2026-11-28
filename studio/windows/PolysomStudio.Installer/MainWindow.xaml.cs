using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Win32;

namespace PolysomStudio.Installer;

public partial class MainWindow : Window
{
    private const string UninstallKeyPath = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\PolysomStudio";
    private const string ShortcutName = "Polysom Studio.lnk";
    private const string SetupShortcutName = "Polysom Studio Setup.lnk";
    private const uint WmSettingChange = 0x001A;
    private const uint SmtoAbortIfHung = 0x0002;
    private static readonly IntPtr HwndBroadcast = new(0xffff);

    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr SendMessageTimeout(
        IntPtr hWnd,
        uint msg,
        UIntPtr wParam,
        string lParam,
        uint fuFlags,
        uint uTimeout,
        out UIntPtr lpdwResult);

    private string? _workspaceRoot;
    private Process? _activeProcess;
    private bool _unattended;
    private bool _noLaunch;
    private readonly string _logPath = Path.Combine(ResolveLogDirectory(), $"studio-setup-{DateTime.Now:yyyyMMdd-HHmmss}.log");

    private static string ResolveLogDirectory()
    {
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var primary = Path.Combine(localAppData, "Lucid Machines", "logs");
        try
        {
            Directory.CreateDirectory(primary);
            return primary;
        }
        catch
        {
            var fallback = Path.Combine(localAppData, "Polysom Studio", "logs");
            Directory.CreateDirectory(fallback);
            return fallback;
        }
    }

    public MainWindow()
    {
        InitializeComponent();
        Loaded += async (_, _) => await InitializeInstallerAsync();
    }

    internal static async Task<int> RunHeadlessCliAsync(string[] args)
    {
        var logPath = Path.Combine(ResolveLogDirectory(), $"studio-setup-{DateTime.Now:yyyyMMdd-HHmmss}.log");
        void Log(string line)
        {
            var clean = StripAnsi(line);
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
                File.AppendAllText(logPath, clean + Environment.NewLine);
            }
            catch (IOException) { }
            Console.WriteLine(clean);
        }

        if (App.HasArg(args, "--smoke-test"))
        {
            return RunSmokeTest(Log);
        }

        var fullUninstall = App.HasArg(args, "--full") || App.HasArg(args, "--remove-env");
        if (App.HasArg(args, "--uninstall"))
        {
            var rootForUninstall = TryFindWorkspaceRoot();
            var scheduledSelfCleanup = PerformUninstall(fullUninstall, rootForUninstall, Log);
            if (scheduledSelfCleanup)
                Log("[Uninstall] Scheduled post-exit cleanup for installed app files.");
            return 0;
        }

        var verifyOnly = App.HasArg(args, "--verify");
        var root = TryFindWorkspaceRoot();
        if (root is null)
        {
            Log("[ERROR] Could not locate Lucid Machines workspace root (pyproject.toml and studio/installer/installer.py).");
            Log("        Extract the PolysomStudio-<version>-win-x64.zip archive completely or install via 'pip install lucid-machines'.");
            return 1;
        }

        var cudaChoice = GetArgValue(args, "--cuda") ?? "auto";
        var skipTorch = App.HasArg(args, "--skip-torch");
        var noShortcuts = verifyOnly || App.HasArg(args, "--no-shortcuts");
        var noPath = verifyOnly || App.HasArg(args, "--no-path");

        var py = TryFindBootstrapPython(root);
        if (py is null && !verifyOnly)
        {
            py = await TryBootstrapPythonViaWingetAsync(Log, root);
        }
        if (py is null)
        {
            LogMissingPythonGuidance(Log, root);
            return 1;
        }

        if (!verifyOnly && !noShortcuts)
        {
            await EnsureWindowsShortcutsCoreAsync(root, Log);
        }

        var exitCode = await ExecutePythonInstallerAsync(
            root,
            py,
            cudaChoice,
            verifyOnly,
            skipTorch,
            noShortcuts,
            noPath,
            Log,
            proc => { });

        if (exitCode == 0 && !verifyOnly)
        {
            RegisterUninstallEntryCore(root, Log);
        }

        return exitCode;
    }

    private static string? GetArgValue(string[] args, string key)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (args[i].Equals(key, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }

    private static int RunSmokeTest(Action<string> log)
    {
        try
        {
            var iconInfo = Application.GetResourceStream(new Uri("pack://application:,,,/polysom.ico"));
            if (iconInfo?.Stream is null || iconInfo.Stream.Length == 0)
            {
                log("[SMOKE FAIL] Embedded polysom.ico resource is missing or empty.");
                return 1;
            }
            iconInfo.Stream.Dispose();

            var root = TryFindWorkspaceRoot();
            if (root is null)
            {
                log("[SMOKE FAIL] TryFindWorkspaceRoot() returned null (missing pyproject.toml / studio/installer/installer.py).");
                return 1;
            }

            var indexHtml = Path.Combine(root, "studio", "src", "index.html");
            var appJs = Path.Combine(root, "studio", "src", "app.js");
            var installerPy = Path.Combine(root, "studio", "installer", "installer.py");
            if (!File.Exists(indexHtml) || !File.Exists(appJs) || !File.Exists(installerPy))
            {
                log($"[SMOKE FAIL] Required Studio bundle assets missing under '{root}'.");
                return 1;
            }

            var target = ResolveStudioTarget(root);
            if (target is null)
            {
                log($"[SMOKE FAIL] ResolveStudioTarget() could not find PolysomStudio.exe or PolysomStudio.bat under '{root}'.");
                return 1;
            }

            log($"[SMOKE OK] PolysomStudio.Installer | workspace={root} | target={target} | icon=embedded");
            return 0;
        }
        catch (Exception ex)
        {
            log($"[SMOKE FAIL] {ex.Message}");
            return 1;
        }
    }

    private async Task InitializeInstallerAsync()
    {
        var args = App.StartupArgs;
        _unattended = App.HasArg(args, "--yes") || App.HasArg(args, "-y");
        _noLaunch = App.HasArg(args, "--no-launch");
        var verifyOnly = App.HasArg(args, "--verify");
        var smokeTest = App.HasArg(args, "--smoke-test");
        var uninstall = App.HasArg(args, "--uninstall");
        var fullUninstall = App.HasArg(args, "--full") || App.HasArg(args, "--remove-env");

        if (smokeTest)
        {
            Application.Current.Shutdown(RunSmokeTest(AppendLog));
            return;
        }

        if (fullUninstall)
        {
            FullUninstallCheck.IsChecked = true;
        }

        if (uninstall)
        {
            InstallButton.IsEnabled = false;
            VerifyButton.IsEnabled = false;
            ShortcutButton.IsEnabled = false;
            _workspaceRoot = TryFindWorkspaceRoot();
            var scheduledSelfCleanup = PerformUninstall(fullUninstall || FullUninstallCheck.IsChecked == true, _workspaceRoot, AppendLog);
            StatusHeadlineText.Text = fullUninstall
                ? "Polysom Studio shortcuts, registry, PATH, and environment were removed."
                : "Polysom Studio shortcuts, PATH entry, and uninstall entry were removed.";
            if (_unattended || scheduledSelfCleanup) Application.Current.Shutdown(0);
            return;
        }

        _workspaceRoot = TryFindWorkspaceRoot();
        if (_workspaceRoot is null)
        {
            StatusHeadlineText.Text = "Could not locate Lucid Machines workspace.";
            WorkspacePathText.Text = "Workspace not found — extract PolysomStudio ZIP or run from repository root.";
            AppendLog("[ERROR] Could not locate pyproject.toml and studio/installer/installer.py.");
            AppendLog("        If you downloaded PolysomStudioSetup.exe standalone, extract PolysomStudio-<version>-win-x64.zip first or run: pip install lucid-machines");
            InstallButton.IsEnabled = false;
            VerifyButton.IsEnabled = false;
            ShortcutButton.IsEnabled = false;
            if (_unattended || verifyOnly) Application.Current.Shutdown(1);
            return;
        }

        WorkspacePathText.Text = $"Workspace: {_workspaceRoot}";
        await ProbeHardwareSummaryAsync();

        var existingPy = Path.Combine(_workspaceRoot, "studio", "studio_env", "Scripts", "python.exe");
        if (File.Exists(existingPy))
        {
            LaunchStudioButton.IsEnabled = true;
            SkipTorchCheck.IsChecked = true;
            AppendLog($"[Detected] Existing managed environment found at: {existingPy} (Fast repair mode enabled)");
        }

        // Never mutate shortcuts during --verify or --smoke-test
        if (!verifyOnly && !smokeTest && DesktopShortcutCheck.IsChecked == true)
        {
            await EnsureWindowsShortcutsAsync();
        }

        if (verifyOnly)
        {
            Application.Current.Shutdown(await RunInstallerProcessAsync(verifyOnly: true));
        }
        else if (_unattended)
        {
            var code = await RunInstallerProcessAsync(verifyOnly: false);
            if (code == 0 && !_noLaunch) LaunchStudio();
            Application.Current.Shutdown(code);
        }
    }

    private async Task ProbeHardwareSummaryAsync()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=name,memory.total --format=csv,noheader")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            using var proc = Process.Start(psi);
            if (proc is not null)
            {
                var output = (await proc.StandardOutput.ReadToEndAsync()).Trim();
                await proc.WaitForExitAsync();
                if (proc.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                {
                    var firstLine = output.Split('\n')[0].Trim();
                    GpuBadgeText.Text = $"GPU: {firstLine}";
                    AppendLog($"[Hardware Probe] NVIDIA GPU detected: {firstLine}");
                    return;
                }
            }
        }
        catch
        {
            // Fallback to CPU badge
        }

        GpuBadgeText.Text = "CPU Mode (No NVIDIA GPU)";
        AppendLog("[Hardware Probe] No NVIDIA GPU detected via nvidia-smi; CPU execution profile available.");
    }

    internal static string? TryFindWorkspaceRoot()
    {
        static bool IsWorkspaceDir(string candidate) =>
            File.Exists(Path.Combine(candidate, "pyproject.toml")) &&
            File.Exists(Path.Combine(candidate, "studio", "installer", "installer.py"));

        var seeds = new[]
        {
            Path.GetDirectoryName(Environment.ProcessPath),
            AppContext.BaseDirectory,
            Directory.GetCurrentDirectory()
        }
        .Where(s => !string.IsNullOrWhiteSpace(s))
        .Distinct(StringComparer.OrdinalIgnoreCase);

        foreach (var seed in seeds)
        {
            for (var dir = new DirectoryInfo(seed!); dir is not null; dir = dir.Parent)
            {
                if (IsWorkspaceDir(dir.FullName))
                    return dir.FullName;

                var appChild = Path.Combine(dir.FullName, "app");
                if (IsWorkspaceDir(appChild))
                    return appChild;
            }
        }

        return null;
    }

    private static bool IsWindowsAppsStub(string path) =>
        path.Contains(@"\Microsoft\WindowsApps\", StringComparison.OrdinalIgnoreCase);

    private static bool IsValidPythonCandidate(string exePath, bool isPyLauncher)
    {
        if (string.IsNullOrWhiteSpace(exePath) || !File.Exists(exePath) || IsWindowsAppsStub(exePath))
            return false;

        try
        {
            var psi = new ProcessStartInfo(exePath)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            if (isPyLauncher) psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add("import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)");

            using var proc = Process.Start(psi);
            if (proc is null) return false;
            if (!proc.WaitForExit(5000))
            {
                try { proc.Kill(); } catch { }
                return false;
            }
            return proc.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private static IEnumerable<string> EnumerateRegistryPythons()
    {
        var results = new List<string>();
        var versions = new[] { "3.14", "3.13", "3.12", "3.11", "3.10" };
        foreach (var hive in new[] { Registry.CurrentUser, Registry.LocalMachine })
        {
            foreach (var ver in versions)
            {
                try
                {
                    using var key = hive.OpenSubKey($@"SOFTWARE\Python\PythonCore\{ver}\InstallPath");
                    if (key is null) continue;
                    var execPath = key.GetValue("ExecutablePath") as string;
                    if (!string.IsNullOrWhiteSpace(execPath))
                        results.Add(execPath);
                    var installDir = key.GetValue(null) as string;
                    if (!string.IsNullOrWhiteSpace(installDir))
                        results.Add(Path.Combine(installDir, "python.exe"));
                }
                catch { }
            }
        }
        return results;
    }

    internal static string? TryFindBootstrapPython(string? root)
    {
        var envCandidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(root))
        {
            envCandidates.Add(Path.Combine(root, "studio", "studio_env", "Scripts", "python.exe"));
            envCandidates.Add(Path.Combine(root, ".venv", "Scripts", "python.exe"));
        }

        foreach (var cand in envCandidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (IsValidPythonCandidate(cand, isPyLauncher: false))
                return cand;
        }

        var pyLocation = Environment.GetEnvironmentVariable("pythonLocation")?.Trim().Trim('"');
        if (!string.IsNullOrWhiteSpace(pyLocation))
        {
            var locCand = Path.Combine(pyLocation, "python.exe");
            if (IsValidPythonCandidate(locCand, isPyLauncher: false))
                return locCand;
        }

        var pathEntries = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries) ?? [];
        foreach (var cmd in new[] { "python.exe", "py.exe" })
        {
            var isPy = cmd.Equals("py.exe", StringComparison.OrdinalIgnoreCase);
            foreach (var folder in pathEntries)
            {
                var trimmed = folder.Trim().Trim('"');
                if (string.IsNullOrWhiteSpace(trimmed)) continue;
                var candidate = Path.Combine(trimmed, cmd);
                if (IsValidPythonCandidate(candidate, isPyLauncher: isPy))
                    return candidate;
            }
        }

        var fallbackCandidates = new List<string>();
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

        foreach (var minor in new[] { "314", "313", "312", "311", "310" })
        {
            if (!string.IsNullOrWhiteSpace(localAppData))
                fallbackCandidates.Add(Path.Combine(localAppData, "Programs", "Python", $"Python{minor}", "python.exe"));
            if (!string.IsNullOrWhiteSpace(programFiles))
                fallbackCandidates.Add(Path.Combine(programFiles, $"Python{minor}", "python.exe"));
            if (!string.IsNullOrWhiteSpace(programFilesX86))
                fallbackCandidates.Add(Path.Combine(programFilesX86, $"Python{minor}", "python.exe"));
        }

        fallbackCandidates.AddRange(EnumerateRegistryPythons());

        foreach (var cand in fallbackCandidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (IsValidPythonCandidate(cand, isPyLauncher: false))
                return cand;
        }

        return null;
    }

    private static string? FindWingetExecutable()
    {
        var pathEntries = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries) ?? [];
        foreach (var folder in pathEntries)
        {
            var candidate = Path.Combine(folder.Trim().Trim('"'), "winget.exe");
            if (File.Exists(candidate)) return candidate;
        }
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var windowsAppsWinget = Path.Combine(localAppData, "Microsoft", "WindowsApps", "winget.exe");
        return File.Exists(windowsAppsWinget) ? windowsAppsWinget : null;
    }

    private static async Task<string?> TryBootstrapPythonViaWingetAsync(Action<string> log, string? workspaceRoot)
    {
        var winget = FindWingetExecutable();
        if (winget is null)
        {
            log("[Bootstrap] Python 3.10+ not found and winget is unavailable on this system.");
            return null;
        }

        log("[Bootstrap] Python 3.10+ was not detected. Attempting automatic user-scoped Python 3.12 installation via winget...");
        try
        {
            var psi = new ProcessStartInfo(winget)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            foreach (var arg in new[]
            {
                "install", "--id", "Python.Python.3.12", "--exact", "--scope", "user",
                "--silent", "--accept-source-agreements", "--accept-package-agreements"
            })
            {
                psi.ArgumentList.Add(arg);
            }

            using var proc = new Process { StartInfo = psi };
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) log($"  [winget] {e.Data}"); };
            proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) log($"  [winget] {e.Data}"); };
            if (!proc.Start())
                return null;
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            await proc.WaitForExitAsync();

            var localPy312 = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Python", "Python312", "python.exe");
            if (IsValidPythonCandidate(localPy312, isPyLauncher: false))
                return localPy312;

            return TryFindBootstrapPython(workspaceRoot);
        }
        catch (Exception ex)
        {
            log($"[Bootstrap Notice] winget invocation failed: {ex.Message}");
            return null;
        }
    }

    private static void LogMissingPythonGuidance(Action<string> log, string? workspaceRoot)
    {
        log("[ERROR] Python 3.10+ could not be located or automatically provisioned.");
        log("        To resolve this:");
        log("        1. Run: winget install --id Python.Python.3.12 --exact --scope user");
        log("        2. Or install Python 3.10–3.12 from https://www.python.org/downloads/ (check 'Add python.exe to PATH')");
        if (!string.IsNullOrWhiteSpace(workspaceRoot))
        {
            var batPath = Path.Combine(workspaceRoot, "studio", "installer", "install_studio.bat");
            if (File.Exists(batPath))
                log($"        3. Or run the fallback script: \"{batPath}\"");
        }
        log("        Support: contact@lucidmachines.co | https://lucidmachines.co");
    }

    private void ResetStageIndicators()
    {
        var muted = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#9CA3AF"));
        Stage1Text.Foreground = muted;
        Stage2Text.Foreground = muted;
        Stage3Text.Foreground = muted;
        Stage4Text.Foreground = muted;
        Stage5Text.Foreground = muted;
        Stage6Text.Foreground = muted;
        Stage1Text.Text = "○  [1/6] Hardware Stack & Driver Probing";
        Stage2Text.Text = "○  [2/6] Isolated Studio Virtual Environment";
        Stage3Text.Text = "○  [3/6] PyTorch & CUDA Tensor Core Runtime";
        Stage4Text.Text = "○  [4/6] Polysom Studio API, WebSockets & Triune Engine";
        Stage5Text.Text = "○  [5/6] Native Shortcuts & Global CLI PATH";
        Stage6Text.Text = "○  [6/6] End-to-End Subsystem Health Check";
    }

    private void UpdateStageProgress(int step, int total)
    {
        var activeBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#38BDF8"));
        var doneBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#10B981"));
        var stages = new[] { Stage1Text, Stage2Text, Stage3Text, Stage4Text, Stage5Text, Stage6Text };
        var labels = new[]
        {
            "[1/6] Hardware Stack & Driver Probing",
            "[2/6] Isolated Studio Virtual Environment",
            "[3/6] PyTorch & CUDA Tensor Core Runtime",
            "[4/6] Polysom Studio API, WebSockets & Triune Engine",
            "[5/6] Native Shortcuts & Global CLI PATH",
            "[6/6] End-to-End Subsystem Health Check"
        };

        var mappedIdx = total == 5 && step == 2 ? 5 : Math.Clamp(step - 1, 0, 5);
        for (var i = 0; i < stages.Length; i++)
        {
            if (i < mappedIdx)
            {
                stages[i].Text = $"✓  {labels[i]}";
                stages[i].Foreground = doneBrush;
            }
            else if (i == mappedIdx)
            {
                stages[i].Text = $"●  {labels[i]}";
                stages[i].Foreground = activeBrush;
            }
        }

        var pct = Math.Clamp((int)Math.Round((step - 0.3) * 100.0 / Math.Max(1, total)), 5, 95);
        InstallProgressBar.Value = pct;
        PercentText.Text = $"{pct}%";
    }

    private void MarkAllStagesComplete()
    {
        var doneBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#10B981"));
        var stages = new[] { Stage1Text, Stage2Text, Stage3Text, Stage4Text, Stage5Text, Stage6Text };
        var labels = new[]
        {
            "[1/6] Hardware Stack & Driver Probing",
            "[2/6] Isolated Studio Virtual Environment",
            "[3/6] PyTorch & CUDA Tensor Core Runtime",
            "[4/6] Polysom Studio API, WebSockets & Triune Engine",
            "[5/6] Native Shortcuts & Global CLI PATH",
            "[6/6] End-to-End Subsystem Health Check"
        };
        for (var i = 0; i < stages.Length; i++)
        {
            stages[i].Text = $"✓  {labels[i]}";
            stages[i].Foreground = doneBrush;
        }
        InstallProgressBar.Value = 100;
        PercentText.Text = "100%";
    }

    private static string StripAnsi(string input)
    {
        return Regex.Replace(input, @"\x1B\[[0-9;]*[mK]", string.Empty);
    }

    private void AppendLog(string line)
    {
        Dispatcher.Invoke(() =>
        {
            var clean = StripAnsi(line);
            try { Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!); File.AppendAllText(_logPath, clean + Environment.NewLine); } catch (IOException) { }
            ConsoleOutputBox.AppendText(clean + Environment.NewLine);
            ConsoleOutputBox.ScrollToEnd();

            var match = Regex.Match(clean, @"^\[(\d+)/(\d+)\]\s+(.+)$");
            if (match.Success && int.TryParse(match.Groups[1].Value, out var step) && int.TryParse(match.Groups[2].Value, out var total))
            {
                StatusHeadlineText.Text = match.Groups[3].Value;
                UpdateStageProgress(step, total);
            }
        });
    }

    private static async Task<int> ExecutePythonInstallerAsync(
        string workspaceRoot,
        string pythonExe,
        string selectedCuda,
        bool verifyOnly,
        bool skipTorch,
        bool noShortcuts,
        bool noPath,
        Action<string> log,
        Action<Process?> trackProcess)
    {
        var script = Path.Combine(workspaceRoot, "studio", "installer", "installer.py");
        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            WorkingDirectory = workspaceRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };
        psi.Environment["PYTHONUTF8"] = "1";
        psi.Environment["PYTHONIOENCODING"] = "utf-8";

        if (Path.GetFileNameWithoutExtension(pythonExe).Equals("py", StringComparison.OrdinalIgnoreCase))
            psi.ArgumentList.Add("-3");

        psi.ArgumentList.Add(script);
        psi.ArgumentList.Add("--yes");
        psi.ArgumentList.Add("--no-launch");
        psi.ArgumentList.Add("--cuda");
        psi.ArgumentList.Add(selectedCuda);

        if (verifyOnly)
        {
            psi.ArgumentList.Add("--verify");
            psi.ArgumentList.Add("--no-shortcuts");
            psi.ArgumentList.Add("--no-path");
        }
        else
        {
            if (skipTorch)
                psi.ArgumentList.Add("--skip-torch");
            if (noShortcuts)
                psi.ArgumentList.Add("--no-shortcuts");
            if (noPath)
                psi.ArgumentList.Add("--no-path");
        }

        using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        trackProcess(proc);
        proc.OutputDataReceived += (_, e) => { if (e.Data is not null) log(e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) log(e.Data); };

        if (!proc.Start())
            throw new InvalidOperationException("Failed to start Python installer process.");

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync();
        var exitCode = proc.ExitCode;
        trackProcess(null);
        return exitCode;
    }

    private async Task<int> RunInstallerProcessAsync(bool verifyOnly)
    {
        if (_workspaceRoot is null) return 1;

        InstallButton.IsEnabled = false;
        VerifyButton.IsEnabled = false;
        ConsoleOutputBox.Clear();
        ResetStageIndicators();
        InstallProgressBar.Value = 5;
        PercentText.Text = "5%";
        StatusHeadlineText.Text = verifyOnly ? "Verifying Polysom Studio installation health..." : "Provisioning Polysom Studio environment...";

        try
        {
            var py = TryFindBootstrapPython(_workspaceRoot);
            if (py is null && !verifyOnly)
            {
                StatusHeadlineText.Text = "Python 3.10+ not detected; bootstrapping via winget...";
                py = await TryBootstrapPythonViaWingetAsync(AppendLog, _workspaceRoot);
            }
            if (py is null)
            {
                LogMissingPythonGuidance(AppendLog, _workspaceRoot);
                StatusHeadlineText.Text = "Python 3.10+ is required before setup can continue.";
                return 1;
            }

            var selectedCuda = (CudaComboBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "auto";
            var noShortcuts = verifyOnly || DesktopShortcutCheck.IsChecked == false;
            var noPath = verifyOnly || RegisterPathCheck.IsChecked == false;

            if (!verifyOnly && DesktopShortcutCheck.IsChecked == true)
            {
                await EnsureWindowsShortcutsAsync();
            }

            var exitCode = await ExecutePythonInstallerAsync(
                _workspaceRoot,
                py,
                selectedCuda,
                verifyOnly,
                SkipTorchCheck.IsChecked == true,
                noShortcuts,
                noPath,
                AppendLog,
                proc => _activeProcess = proc);

            if (exitCode == 0)
            {
                if (!verifyOnly) RegisterUninstallEntryCore(_workspaceRoot, AppendLog);
                MarkAllStagesComplete();
                StatusHeadlineText.Text = verifyOnly
                    ? "All Polysom Studio & Lucid Machines subsystems passed verification!"
                    : "Polysom Studio & CLI installed and verified successfully!";
                LaunchStudioButton.IsEnabled = true;
            }
            else
            {
                StatusHeadlineText.Text = $"Setup exited with code {exitCode}. Review the console output below.";
            }

            return exitCode;
        }
        catch (Exception ex)
        {
            StatusHeadlineText.Text = "Setup encountered an error.";
            AppendLog($"[ERROR] {ex.Message}");
            return 1;
        }
        finally
        {
            InstallButton.IsEnabled = true;
            VerifyButton.IsEnabled = true;
        }
    }

    internal static string? ResolveStudioTarget(string root) => new[]
    {
        Path.Combine(AppContext.BaseDirectory, "PolysomStudio.exe"),
        Path.Combine(AppContext.BaseDirectory, "app", "PolysomStudio.exe"),
        Path.Combine(root, "app", "PolysomStudio.exe"),
        Path.Combine(root, "PolysomStudio.exe"),
        Path.Combine(root, "studio", "publish", "app", "PolysomStudio.exe"),
        Path.Combine(root, "studio", "publish", "PolysomStudio.exe"),
        Path.Combine(root, "studio", "windows", "publish", "PolysomStudio.exe"),
        Path.Combine(root, "studio", "windows", "PolysomStudio.Windows", "bin", "Release", "net10.0-windows", "PolysomStudio.exe"),
        Path.Combine(root, "studio", "windows", "PolysomStudio.Windows", "bin", "Release", "net10.0-windows", "win-x64", "PolysomStudio.exe"),
        Path.Combine(root, "studio", "PolysomStudio.bat")
    }.FirstOrDefault(File.Exists);

    private static string[] DesktopDirectories()
    {
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return new[]
        {
            Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
            Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
            Path.Combine(userProfile, "OneDrive", "Desktop"),
            Path.Combine(userProfile, "Desktop")
        }
        .Where(d => !string.IsNullOrWhiteSpace(d) && Directory.Exists(d))
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToArray();
    }

    private static void RegisterUninstallEntryCore(string workspaceRoot, Action<string> log)
    {
        if (Environment.ProcessPath is not { } setupExe) return;
        var studioExe = ResolveStudioTarget(workspaceRoot);
        using var key = Registry.CurrentUser.CreateSubKey(UninstallKeyPath);
        key.SetValue("DisplayName", "Polysom Studio");
        key.SetValue("DisplayVersion", typeof(App).Assembly.GetName().Version?.ToString(3) ?? "0.0.0");
        key.SetValue("Publisher", "Lucid Machines");
        key.SetValue("Contact", "contact@lucidmachines.co");
        key.SetValue("HelpLink", "https://lucidmachines.co");
        key.SetValue("URLInfoAbout", "https://lucidmachines.co");
        key.SetValue("InstallLocation", workspaceRoot);
        key.SetValue("DisplayIcon", studioExe ?? setupExe);
        key.SetValue("UninstallString", $"\"{setupExe}\" --uninstall");
        key.SetValue("QuietUninstallString", $"\"{setupExe}\" --uninstall --yes");
        key.SetValue("NoModify", 1, RegistryValueKind.DWord);
        key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
        log("[Registry] Registered Apps & Features uninstall entry (HKCU).");
    }

    private static void RemoveScriptsFromUserPath(string? installRoot, Action<string> log)
    {
        try
        {
            using var envKey = Registry.CurrentUser.OpenSubKey("Environment", writable: true);
            if (envKey is null) return;
            var current = envKey.GetValue("Path", "", RegistryValueOptions.DoNotExpandEnvironmentNames) as string ?? "";
            if (string.IsNullOrWhiteSpace(current)) return;

            var targetScripts = !string.IsNullOrWhiteSpace(installRoot)
                ? Path.GetFullPath(Path.Combine(installRoot, "studio", "studio_env", "Scripts"))
                : null;

            var parts = current.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var kept = new List<string>(parts.Length);
            var removedCount = 0;

            foreach (var part in parts)
            {
                var isMatch = false;
                try
                {
                    var normPart = Path.GetFullPath(Environment.ExpandEnvironmentVariables(part));
                    if (targetScripts is not null && normPart.Equals(targetScripts, StringComparison.OrdinalIgnoreCase))
                        isMatch = true;
                    else if (normPart.EndsWith(Path.Combine("studio", "studio_env", "Scripts"), StringComparison.OrdinalIgnoreCase))
                        isMatch = true;
                }
                catch
                {
                    if (part.EndsWith(@"studio\studio_env\Scripts", StringComparison.OrdinalIgnoreCase))
                        isMatch = true;
                }

                if (isMatch) removedCount++;
                else kept.Add(part);
            }

            if (removedCount > 0)
            {
                envKey.SetValue("Path", string.Join(";", kept), RegistryValueKind.ExpandString);
                SendMessageTimeout(HwndBroadcast, WmSettingChange, UIntPtr.Zero, "Environment", SmtoAbortIfHung, 2000, out _);
                log($"[Uninstall] Removed {removedCount} Studio CLI entry from Windows User PATH.");
            }
        }
        catch (Exception ex)
        {
            log($"[Uninstall Notice] Could not clean User PATH: {ex.Message}");
        }
    }

    private static bool HasProtectedRepoOrUserData(string installRoot)
    {
        foreach (var marker in new[] { ".git", "checkpoints", "data", "exports" })
        {
            if (Directory.Exists(Path.Combine(installRoot, marker)) || File.Exists(Path.Combine(installRoot, marker)))
                return true;
        }
        return false;
    }

    private static void StopRunningStudioProcesses(string? installRoot, Action<string> log)
    {
        if (string.IsNullOrWhiteSpace(installRoot)) return;
        var normRoot = Path.GetFullPath(installRoot).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        foreach (var proc in Process.GetProcessesByName("PolysomStudio"))
        {
            try
            {
                var exePath = proc.MainModule?.FileName;
                if (!string.IsNullOrWhiteSpace(exePath) &&
                    Path.GetFullPath(exePath).StartsWith(normRoot, StringComparison.OrdinalIgnoreCase))
                {
                    proc.Kill(entireProcessTree: true);
                    proc.WaitForExit(3000);
                    log($"[Uninstall] Stopped running PolysomStudio.exe (PID {proc.Id}).");
                }
            }
            catch { }
            finally { proc.Dispose(); }
        }
    }

    private static bool PerformUninstall(bool fullUninstall, string? workspaceRoot, Action<string> log)
    {
        string? installRoot = workspaceRoot;
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(UninstallKeyPath);
            if (string.IsNullOrWhiteSpace(installRoot))
                installRoot = key?.GetValue("InstallLocation") as string;
        }
        catch { }

        var removed = 0;
        var links = DesktopDirectories()
            .Select(d => Path.Combine(d, ShortcutName))
            .Concat(new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), ShortcutName),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), SetupShortcutName)
            });
        foreach (var link in links.Where(File.Exists))
        {
            try
            {
                File.Delete(link);
                removed++;
            }
            catch { }
        }
        Registry.CurrentUser.DeleteSubKeyTree(UninstallKeyPath, throwOnMissingSubKey: false);
        RemoveScriptsFromUserPath(installRoot, log);
        log($"[Uninstall] Removed {removed} shortcut(s) and HKCU uninstall registry entry.");

        if (!fullUninstall)
        {
            log("[Uninstall] Program files and the managed Python environment were preserved (pass --full to remove them).");
            return false;
        }

        StopRunningStudioProcesses(installRoot, log);

        if (!string.IsNullOrWhiteSpace(installRoot))
        {
            var studioEnvDir = Path.Combine(installRoot, "studio", "studio_env");
            if (Directory.Exists(studioEnvDir))
            {
                try
                {
                    Directory.Delete(studioEnvDir, recursive: true);
                    log($"[Uninstall] Removed managed virtual environment: {studioEnvDir}");
                }
                catch (Exception ex)
                {
                    log($"[Uninstall Notice] Could not remove {studioEnvDir}: {ex.Message}");
                }
            }
        }

        var webViewData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "LucidMachines", "PolysomStudio", "WebView2");
        if (Directory.Exists(webViewData))
        {
            try
            {
                Directory.Delete(webViewData, recursive: true);
                log($"[Uninstall] Removed WebView2 cache: {webViewData}");
            }
            catch (Exception ex)
            {
                log($"[Uninstall Notice] Could not remove WebView2 cache: {ex.Message}");
            }
        }

        if (string.IsNullOrWhiteSpace(installRoot) || !Directory.Exists(installRoot))
            return false;

        if (HasProtectedRepoOrUserData(installRoot))
        {
            log("[Uninstall] Protected repository or user data (.git/checkpoints/data/exports) detected; preserving source/workspace files.");
            return false;
        }

        var cleanupTargets = new List<string>();
        var appDir = Path.Combine(installRoot, "app");
        if (Directory.Exists(appDir)) cleanupTargets.Add($"rmdir /s /q \"{appDir}\"");

        if (Environment.ProcessPath is { } procExe &&
            Path.GetFullPath(procExe).StartsWith(Path.GetFullPath(installRoot), StringComparison.OrdinalIgnoreCase))
        {
            cleanupTargets.Add($"del /f /q \"{procExe}\"");
        }

        if (cleanupTargets.Count == 0) return false;

        try
        {
            var cmdChain = string.Join(" & ", cleanupTargets);
            Process.Start(new ProcessStartInfo("cmd.exe", $"/c ping -n 3 127.0.0.1 >nul & {cmdChain}")
            {
                WorkingDirectory = Path.GetTempPath(),
                CreateNoWindow = true,
                UseShellExecute = false
            });
            return true;
        }
        catch (Exception ex)
        {
            log($"[Uninstall Notice] Could not schedule post-exit cleanup: {ex.Message}");
            return false;
        }
    }

    private Task EnsureWindowsShortcutsAsync() =>
        _workspaceRoot is null ? Task.CompletedTask : EnsureWindowsShortcutsCoreAsync(_workspaceRoot, AppendLog);

    private static async Task EnsureWindowsShortcutsCoreAsync(string workspaceRoot, Action<string> log)
    {
        try
        {
            var studioExe = ResolveStudioTarget(workspaceRoot) ?? Path.Combine(workspaceRoot, "app", "PolysomStudio.exe");
            var setupExe = Environment.ProcessPath ?? Path.Combine(workspaceRoot, "PolysomStudioSetup.exe");
            var iconPath = Path.Combine(workspaceRoot, "studio", "windows", "polysom.ico");
            var iconTarget = File.Exists(iconPath) ? iconPath : studioExe;

            var desktopDirs = DesktopDirectories();
            var startMenuDir = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
            var sb = new StringBuilder();
            sb.AppendLine("$WshShell = New-Object -comObject WScript.Shell");
            foreach (var dir in desktopDirs)
            {
                var lnk = Path.Combine(dir, ShortcutName);
                try { if (File.Exists(lnk)) File.Delete(lnk); } catch { }
                sb.AppendLine($"$s = $WshShell.CreateShortcut('{lnk.Replace("'", "''")}');");
                sb.AppendLine($"$s.TargetPath = '{studioExe.Replace("'", "''")}';");
                sb.AppendLine($"$s.WorkingDirectory = '{workspaceRoot.Replace("'", "''")}';");
                sb.AppendLine($"$s.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                sb.AppendLine("$s.Description = 'Polysom Studio - Lucid Machines AI Engine and Visual Research IDE';");
                sb.AppendLine("$s.Save();");
            }

            if (!string.IsNullOrWhiteSpace(startMenuDir) && Directory.Exists(startMenuDir))
            {
                var startLnk = Path.Combine(startMenuDir, ShortcutName);
                try { if (File.Exists(startLnk)) File.Delete(startLnk); } catch { }
                sb.AppendLine($"$sm = $WshShell.CreateShortcut('{startLnk.Replace("'", "''")}');");
                sb.AppendLine($"$sm.TargetPath = '{studioExe.Replace("'", "''")}';");
                sb.AppendLine($"$sm.WorkingDirectory = '{workspaceRoot.Replace("'", "''")}';");
                sb.AppendLine($"$sm.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                sb.AppendLine("$sm.Description = 'Polysom Studio - Lucid Machines AI Engine and Visual Research IDE';");
                sb.AppendLine("$sm.Save();");

                if (File.Exists(setupExe))
                {
                    var setupLnk = Path.Combine(startMenuDir, SetupShortcutName);
                    try { if (File.Exists(setupLnk)) File.Delete(setupLnk); } catch { }
                    sb.AppendLine($"$su = $WshShell.CreateShortcut('{setupLnk.Replace("'", "''")}');");
                    sb.AppendLine($"$su.TargetPath = '{setupExe.Replace("'", "''")}';");
                    sb.AppendLine($"$su.WorkingDirectory = '{workspaceRoot.Replace("'", "''")}';");
                    sb.AppendLine($"$su.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                    sb.AppendLine("$su.Description = 'Polysom Studio Setup - Repair or Reconfigure Lucid Machines Engine';");
                    sb.AppendLine("$su.Save();");
                }
            }

            var psi = new ProcessStartInfo("powershell.exe")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            psi.ArgumentList.Add("-NoProfile");
            psi.ArgumentList.Add("-Command");
            psi.ArgumentList.Add(sb.ToString());

            using var proc = Process.Start(psi);
            if (proc is not null)
            {
                await proc.WaitForExitAsync();
                if (proc.ExitCode == 0)
                {
                    log($"[Shortcuts] Created Desktop & Start Menu shortcuts ({string.Join(", ", desktopDirs)}) -> {Path.GetFileName(studioExe)}");
                }
            }
        }
        catch (Exception ex)
        {
            log($"[Shortcuts Notice] {ex.Message}");
        }
    }

    private async void ShortcutButton_Click(object sender, RoutedEventArgs e)
    {
        await EnsureWindowsShortcutsAsync();
        StatusHeadlineText.Text = "Desktop and Start Menu shortcuts created for Polysom Studio.";
    }

    private async void InstallButton_Click(object sender, RoutedEventArgs e)
    {
        await RunInstallerProcessAsync(verifyOnly: false);
    }

    private async void VerifyButton_Click(object sender, RoutedEventArgs e)
    {
        await RunInstallerProcessAsync(verifyOnly: true);
    }

    private void UninstallButton_Click(object sender, RoutedEventArgs e)
    {
        var full = FullUninstallCheck.IsChecked == true;
        var scheduledSelfCleanup = PerformUninstall(full, _workspaceRoot, AppendLog);
        StatusHeadlineText.Text = full
            ? "Polysom Studio shortcuts, registry, PATH, and environment were removed."
            : "Polysom Studio shortcuts, PATH entry, and uninstall entry were removed.";
        if (scheduledSelfCleanup)
        {
            Application.Current.Shutdown(0);
        }
    }

    private void LaunchStudioButton_Click(object sender, RoutedEventArgs e)
    {
        if (LaunchStudio()) Close();
    }

    private bool LaunchStudio()
    {
        if (_workspaceRoot is null) return false;
        var target = ResolveStudioTarget(_workspaceRoot);

        if (target is null)
        {
            AppendLog("[ERROR] Could not locate PolysomStudio.exe.");
            return false;
        }

        Process.Start(new ProcessStartInfo
        {
            FileName = target,
            WorkingDirectory = _workspaceRoot,
            UseShellExecute = true
        });
        return true;
    }

    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            if (_activeProcess is { HasExited: false })
            {
                _activeProcess.Kill(entireProcessTree: true);
            }
        }
        catch
        {
        }
    }
}
