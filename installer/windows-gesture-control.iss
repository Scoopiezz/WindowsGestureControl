; Inno Setup Script for Windows Gesture Control
; Generated for v0.9.0 release
; Installer output: WindowsGestureControlSetup.exe

[Setup]
AppName=Windows Gesture Control
AppVersion=0.9.0
AppPublisher=Windows Gesture Control Project
AppPublisherURL=https://github.com
AppSupportURL=https://github.com
AppUpdatesURL=https://github.com
DefaultDirName={autopf}\WindowsGestureControl
DefaultGroupName=WindowsGestureControl
AllowNoIcons=yes
LicenseFile=..\LICENSE.txt
InfoBeforeFile=..\README.md
OutputDir=..\dist
OutputBaseFilename=WindowsGestureControlSetup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
WizardStyle=modern
UninstallDisplayIcon={app}\WindowsGestureControl-v0.9.0.exe
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\WindowsGestureControl-v0.9.0.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\WindowsGestureControl"; Filename: "{app}\WindowsGestureControl-v0.9.0.exe"
Name: "{group}\{cm:UninstallProgram,WindowsGestureControl}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\WindowsGestureControl"; Filename: "{app}\WindowsGestureControl-v0.9.0.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\WindowsGestureControl-v0.9.0.exe"; Description: "{cm:LaunchProgram,Windows Gesture Control}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"
