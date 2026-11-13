$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$version = (Select-String -Path (Join-Path $repo 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$name = "TriuneStudio-$version-win-x64"
$dist = Join-Path $repo 'dist'
$package = Join-Path $dist $name
$zip = Join-Path $dist "$name.zip"
$publishLocal = Join-Path $repo 'studio/publish'

foreach ($project in 'TriuneStudio.Windows', 'TriuneStudio.Installer') {
    Write-Host "Publishing $project..."
    dotnet publish (Join-Path $repo "studio/windows/$project/$project.csproj") --configuration Release --runtime win-x64 --self-contained true --output $publishLocal
    if ($LASTEXITCODE -ne 0) { throw "$project publish failed with exit code $LASTEXITCODE" }
}

if (Test-Path $package) { Remove-Item $package -Recurse -Force }
if (Test-Path $zip) { Remove-Item $zip -Force }
$null = New-Item -ItemType Directory -Path (Join-Path $package 'scripts'), (Join-Path $package 'studio/windows') -Force

foreach ($exe in 'TriuneStudio.exe', 'TriuneStudioSetup.exe') {
    Copy-Item (Join-Path $publishLocal $exe) $package
}
foreach ($file in 'pyproject.toml', 'README.md', 'requirements.txt') {
    if (Test-Path (Join-Path $repo $file)) { Copy-Item (Join-Path $repo $file) $package }
}
foreach ($dir in 'docs', 'triune', 'models', 'studio/src', 'studio/installer') {
    Copy-Item (Join-Path $repo $dir) (Join-Path $package $dir) -Recurse
}
Copy-Item (Join-Path $repo 'studio/TriuneStudio.bat') (Join-Path $package 'studio')
Copy-Item (Join-Path $repo 'studio/windows/triune.ico') (Join-Path $package 'studio/windows')
foreach ($script in 'launch_studio.py', 'benchmark_inference.py') {
    Copy-Item (Join-Path $repo "scripts/$script") (Join-Path $package 'scripts')
}
Get-ChildItem $package -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip
Remove-Item $package -Recurse -Force
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
Set-Content -Path "$zip.sha256" -Value "$hash  $name.zip"

Write-Host "Package:  $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
Write-Host "SHA256:   $hash"
Write-Host "Binaries: $publishLocal (TriuneStudio.exe, TriuneStudioSetup.exe)"
