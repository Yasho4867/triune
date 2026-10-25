$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$package = Join-Path $repo "dist/TriuneStudio-$stamp"
$publish = Join-Path $package 'studio/windows/publish'
$null = New-Item -ItemType Directory -Path $publish -Force

$project = Join-Path $repo 'studio/windows/TriuneStudio.Windows/TriuneStudio.Windows.csproj'
dotnet publish $project --configuration Release --runtime win-x64 --self-contained true --output $publish
if ($LASTEXITCODE -ne 0) {
    throw "Native Studio publish failed with exit code $LASTEXITCODE"
}

Copy-Item (Join-Path $repo 'pyproject.toml') $package
Copy-Item (Join-Path $repo 'README.md') $package
Copy-Item (Join-Path $repo 'docs') (Join-Path $package 'docs') -Recurse
if (Test-Path (Join-Path $repo 'requirements.txt')) { Copy-Item (Join-Path $repo 'requirements.txt') $package }
Copy-Item (Join-Path $repo 'triune') (Join-Path $package 'triune') -Recurse
Copy-Item (Join-Path $repo 'models') (Join-Path $package 'models') -Recurse
Copy-Item (Join-Path $repo 'studio/src') (Join-Path $package 'studio/src') -Recurse
Copy-Item (Join-Path $repo 'studio/installer') (Join-Path $package 'studio/installer') -Recurse
Copy-Item (Join-Path $repo 'studio/TriuneStudio.bat') (Join-Path $package 'studio/TriuneStudio.bat')

$scripts = Join-Path $package 'scripts'
$null = New-Item -ItemType Directory -Path $scripts -Force
Copy-Item (Join-Path $repo 'scripts/launch_studio.py') $scripts
Copy-Item (Join-Path $repo 'scripts/benchmark_inference.py') $scripts

Compress-Archive -Path (Join-Path $package '*') -DestinationPath "$package.zip"
Write-Host "Triune Studio package created: $package"
Write-Host "Portable package archive: $package.zip"
Write-Host "Launch studio/windows/publish/TriuneStudio.exe. First launch can install the managed Python/CUDA environment."
