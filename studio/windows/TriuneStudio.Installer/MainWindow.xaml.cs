using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace TriuneStudio.Installer;

public partial class MainWindow : Window
{
    private string? _workspaceRoot;
    private Process? _activeProcess;
    private bool _unattended;
    private bool _noLaunch;

    public MainWindow()
    {
        InitializeComponent();
        Loaded += async (_, _) => await InitializeInstallerAsync();
    }

    private async Task InitializeInstallerAsync()
    {
        var args = App.StartupArgs;
        _unattended = args.Any(a => a.Equals("--yes", StringComparison.OrdinalIgnoreCase) || a.Equals("-y", StringComparison.OrdinalIgnoreCase));
        _noLaunch = args.Any(a => a.Equals("--no-launch", StringComparison.OrdinalIgnoreCase));
        var verifyOnly = args.Any(a => a.Equals("--verify", StringComparison.OrdinalIgnoreCase));

        try
        {
            _workspaceRoot = FindWorkspaceRoot();
            WorkspacePathText.Text = $"Workspace: {_workspaceRoot}";
        }
        catch (Exception ex)
        {
            StatusHeadlineText.Text = "Could not locate Triune repository root.";
            AppendLog($"[ERROR] {ex.Message}");
            InstallButton.IsEnabled = false;
            VerifyButton.IsEnabled = false;
            return;
        }

        await ProbeHardwareSummaryAsync();

        // Check if existing studio_env with CUDA PyTorch is already present to default fast-repair option
        var existingPy = Path.Combine(_workspaceRoot, "studio", "studio_env", "Scripts", "python.exe");
        if (File.Exists(existingPy))
        {
            LaunchStudioButton.IsEnabled = true;
            SkipTorchCheck.IsChecked = true;
            AppendLog($"[Detected] Existing managed environment found at: {existingPy} (Fast repair mode enabled)");
        }

        if (DesktopShortcutCheck.IsChecked == true)
        {
            await EnsureWindowsShortcutsAsync();
        }

        if (verifyOnly)
        {
            await RunInstallerProcessAsync(verifyOnly: true);
        }
        else if (_unattended)
        {
            var code = await RunInstallerProcessAsync(verifyOnly: false);
            if (code == 0 && _noLaunch)
            {
                Application.Current.Shutdown(0);
            }
            else if (code != 0 && _noLaunch)
            {
                Application.Current.Shutdown(code);
            }
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

    private static string FindWorkspaceRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml")) &&
                File.Exists(Path.Combine(dir.FullName, "studio", "installer", "installer.py")))
            {
                return dir.FullName;
            }
        }
        var cwd = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (var dir = cwd; dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml")) &&
                File.Exists(Path.Combine(dir.FullName, "studio", "installer", "installer.py")))
            {
                return dir.FullName;
            }
        }
        throw new DirectoryNotFoundException("Could not locate pyproject.toml and studio/installer/installer.py.");
    }

    private static string FindBootstrapPython(string root)
    {
        var candidates = new[]
        {
            Path.Combine(root, "studio", "studio_env", "Scripts", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python314", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python313", "python.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python", "Python312", "python.exe"),
        };
        foreach (var cand in candidates)
        {
            if (File.Exists(cand)) return cand;
        }

        foreach (var cmd in new[] { "python.exe", "py.exe" })
        {
            var path = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator)
                .Select(p => Path.Combine(p.Trim('"'), cmd))
                .FirstOrDefault(File.Exists);
            if (path is not null) return path;
        }

        throw new FileNotFoundException("Python 3.10+ was not found on this machine. Install Python from python.org first.");
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
        Stage4Text.Text = "○  [4/6] Studio API, WebSockets & Triune Engine";
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
            "[4/6] Studio API, WebSockets & Triune Engine",
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
            "[4/6] Studio API, WebSockets & Triune Engine",
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

    private async Task<int> RunInstallerProcessAsync(bool verifyOnly)
    {
        if (_workspaceRoot is null) return 1;

        InstallButton.IsEnabled = false;
        VerifyButton.IsEnabled = false;
        ConsoleOutputBox.Clear();
        ResetStageIndicators();
        InstallProgressBar.Value = 5;
        PercentText.Text = "5%";
        StatusHeadlineText.Text = verifyOnly ? "Verifying Triune Studio installation health..." : "Provisioning Triune Studio environment...";

        try
        {
            var py = FindBootstrapPython(_workspaceRoot);
            var script = Path.Combine(_workspaceRoot, "studio", "installer", "installer.py");
            var selectedCuda = (CudaComboBox.SelectedItem as ComboBoxItem)?.Tag?.ToString() ?? "auto";

            var psi = new ProcessStartInfo
            {
                FileName = py,
                WorkingDirectory = _workspaceRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            psi.Environment["PYTHONUTF8"] = "1";
            psi.Environment["PYTHONIOENCODING"] = "utf-8";

            if (Path.GetFileNameWithoutExtension(py).Equals("py", StringComparison.OrdinalIgnoreCase))
                psi.ArgumentList.Add("-3");

            psi.ArgumentList.Add(script);
            psi.ArgumentList.Add("--yes");
            psi.ArgumentList.Add("--no-launch");
            psi.ArgumentList.Add("--cuda");
            psi.ArgumentList.Add(selectedCuda);

            if (verifyOnly)
            {
                psi.ArgumentList.Add("--verify");
                if (DesktopShortcutCheck.IsChecked == false)
                    psi.ArgumentList.Add("--no-shortcuts");
                if (RegisterPathCheck.IsChecked == false)
                    psi.ArgumentList.Add("--no-path");
            }
            else
            {
                if (SkipTorchCheck.IsChecked == true)
                    psi.ArgumentList.Add("--skip-torch");
                if (DesktopShortcutCheck.IsChecked == false)
                    psi.ArgumentList.Add("--no-shortcuts");
                if (RegisterPathCheck.IsChecked == false)
                    psi.ArgumentList.Add("--no-path");
            }

            if (DesktopShortcutCheck.IsChecked == true)
            {
                await EnsureWindowsShortcutsAsync();
            }

            _activeProcess = new Process { StartInfo = psi, EnableRaisingEvents = true };
            _activeProcess.OutputDataReceived += (_, e) => { if (e.Data is not null) AppendLog(e.Data); };
            _activeProcess.ErrorDataReceived += (_, e) => { if (e.Data is not null) AppendLog(e.Data); };

            if (!_activeProcess.Start())
                throw new InvalidOperationException("Failed to start Python installer process.");

            _activeProcess.BeginOutputReadLine();
            _activeProcess.BeginErrorReadLine();
            await _activeProcess.WaitForExitAsync();

            var exitCode = _activeProcess.ExitCode;
            _activeProcess.Dispose();
            _activeProcess = null;

            if (exitCode == 0)
            {
                MarkAllStagesComplete();
                StatusHeadlineText.Text = verifyOnly
                    ? "All Triune Studio & AI Engine subsystems passed verification!"
                    : "Triune Studio & CLI installed and verified successfully!";
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

    private async Task EnsureWindowsShortcutsAsync()
    {
        if (_workspaceRoot is null) return;
        try
        {
            var studioCandidates = new[]
            {
                Path.Combine(_workspaceRoot, "TriuneStudio.exe"),
                Path.Combine(_workspaceRoot, "studio", "windows", "publish", "TriuneStudio.exe"),
                Path.Combine(_workspaceRoot, "studio", "windows", "TriuneStudio.Windows", "bin", "Release", "net10.0-windows", "TriuneStudio.exe"),
                Path.Combine(_workspaceRoot, "studio", "TriuneStudio.bat")
            };
            var studioExe = studioCandidates.FirstOrDefault(File.Exists) ?? studioCandidates[0];
            var setupExe = Path.Combine(_workspaceRoot, "TriuneStudioSetup.exe");
            var iconPath = Path.Combine(_workspaceRoot, "studio", "windows", "triune.ico");
            var iconTarget = File.Exists(iconPath) ? iconPath : studioExe;

            var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var desktopDirs = new[]
            {
                Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                Path.Combine(userProfile, "OneDrive", "Desktop"),
                Path.Combine(userProfile, "Desktop")
            }
            .Where(d => !string.IsNullOrWhiteSpace(d) && Directory.Exists(d))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

            var startMenuDir = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
            var sb = new StringBuilder();
            sb.AppendLine("$WshShell = New-Object -comObject WScript.Shell");
            foreach (var dir in desktopDirs)
            {
                var lnk = Path.Combine(dir, "Triune Studio.lnk");
                try { if (File.Exists(lnk)) File.Delete(lnk); } catch { }
                sb.AppendLine($"$s = $WshShell.CreateShortcut('{lnk.Replace("'", "''")}');");
                sb.AppendLine($"$s.TargetPath = '{studioExe.Replace("'", "''")}';");
                sb.AppendLine($"$s.WorkingDirectory = '{_workspaceRoot.Replace("'", "''")}';");
                sb.AppendLine($"$s.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                sb.AppendLine("$s.Description = 'Triune Studio - AI Engine and Visual Research IDE';");
                sb.AppendLine("$s.Save();");
            }

            if (!string.IsNullOrWhiteSpace(startMenuDir) && Directory.Exists(startMenuDir))
            {
                var startLnk = Path.Combine(startMenuDir, "Triune Studio.lnk");
                try { if (File.Exists(startLnk)) File.Delete(startLnk); } catch { }
                sb.AppendLine($"$sm = $WshShell.CreateShortcut('{startLnk.Replace("'", "''")}');");
                sb.AppendLine($"$sm.TargetPath = '{studioExe.Replace("'", "''")}';");
                sb.AppendLine($"$sm.WorkingDirectory = '{_workspaceRoot.Replace("'", "''")}';");
                sb.AppendLine($"$sm.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                sb.AppendLine("$sm.Description = 'Triune Studio - AI Engine and Visual Research IDE';");
                sb.AppendLine("$sm.Save();");

                if (File.Exists(setupExe))
                {
                    var setupLnk = Path.Combine(startMenuDir, "Triune Studio Setup.lnk");
                    try { if (File.Exists(setupLnk)) File.Delete(setupLnk); } catch { }
                    sb.AppendLine($"$su = $WshShell.CreateShortcut('{setupLnk.Replace("'", "''")}');");
                    sb.AppendLine($"$su.TargetPath = '{setupExe.Replace("'", "''")}';");
                    sb.AppendLine($"$su.WorkingDirectory = '{_workspaceRoot.Replace("'", "''")}';");
                    sb.AppendLine($"$su.IconLocation = '{iconTarget.Replace("'", "''")},0';");
                    sb.AppendLine("$su.Description = 'Triune Studio Setup - Repair or Reconfigure AI Engine';");
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
                    AppendLog($"[Shortcuts] Created Desktop & Start Menu shortcuts ({string.Join(", ", desktopDirs)}) -> {Path.GetFileName(studioExe)}");
                }
            }
        }
        catch (Exception ex)
        {
            AppendLog($"[Shortcuts Notice] {ex.Message}");
        }
    }

    private async void ShortcutButton_Click(object sender, RoutedEventArgs e)
    {
        await EnsureWindowsShortcutsAsync();
        StatusHeadlineText.Text = "Desktop and Start Menu shortcuts created for Triune Studio.";
    }

    private async void InstallButton_Click(object sender, RoutedEventArgs e)
    {
        await RunInstallerProcessAsync(verifyOnly: false);
    }

    private async void VerifyButton_Click(object sender, RoutedEventArgs e)
    {
        await RunInstallerProcessAsync(verifyOnly: true);
    }

    private void LaunchStudioButton_Click(object sender, RoutedEventArgs e)
    {
        if (_workspaceRoot is null) return;
        var candidates = new[]
        {
            Path.Combine(_workspaceRoot, "TriuneStudio.exe"),
            Path.Combine(_workspaceRoot, "studio", "windows", "publish", "TriuneStudio.exe"),
            Path.Combine(_workspaceRoot, "studio", "windows", "TriuneStudio.Windows", "bin", "Release", "net10.0-windows", "TriuneStudio.exe"),
            Path.Combine(_workspaceRoot, "studio", "TriuneStudio.bat")
        };

        var target = candidates.FirstOrDefault(File.Exists);
        if (target is null)
        {
            AppendLog("[ERROR] Could not locate TriuneStudio.exe.");
            return;
        }

        Process.Start(new ProcessStartInfo
        {
            FileName = target,
            WorkingDirectory = _workspaceRoot,
            UseShellExecute = true
        });
        Close();
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
