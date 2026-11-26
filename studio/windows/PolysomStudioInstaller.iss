; Optional Inno Setup script for Polysom Studio (Lucid Machines)
; Compiled by scripts/publish_studio_windows.ps1 when iscc.exe is available.

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

[Setup]
AppId={{8B4E1A29-9C71-42D1-A512-90E11234AB01}
AppName=Polysom Studio
AppVersion={#MyAppVersion}
AppPublisher=Lucid Machines
AppPublisherURL=https://lucidmachines.co
AppSupportURL=https://github.com/lucid-machines/triune
AppUpdatesURL=https://lucidmachines.co
AppContact=contact@lucidmachines.co
DefaultDirName={localappdata}\LucidMachines\PolysomStudio
DefaultGroupName=Polysom Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=polysom.ico
UninstallDisplayIcon={app}\app\PolysomStudio.exe
OutputBaseFilename=PolysomStudioSetup

[Files]
Source: "..\publish\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\lucid_machines\*"; DestDir: "{app}\lucid_machines"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\..\models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\studio\src\*"; DestDir: "{app}\studio\src"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\studio\installer\*"; DestDir: "{app}\studio\installer"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__,*.pyc"
Source: "..\..\studio\PolysomStudio.bat"; DestDir: "{app}\studio"; Flags: ignoreversion
Source: "polysom.ico"; DestDir: "{app}\studio\windows"; Flags: ignoreversion
Source: "..\..\scripts\launch_studio.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\scripts\benchmark_inference.py"; DestDir: "{app}\scripts"; Flags: ignoreversion

[Icons]
Name: "{group}\Polysom Studio"; Filename: "{app}\app\PolysomStudio.exe"; WorkingDir: "{app}"; IconFilename: "{app}\studio\windows\polysom.ico"
Name: "{userdesktop}\Polysom Studio"; Filename: "{app}\app\PolysomStudio.exe"; WorkingDir: "{app}"; IconFilename: "{app}\studio\windows\polysom.ico"

[Run]
Filename: "{app}\app\PolysomStudioSetup.exe"; Parameters: "--yes --no-launch"; Flags: nowait postinstall skipifsilent; Description: "Provision Python/CUDA Studio runtime & register CLI tools (kcx, polysom, lucid-machines)"
