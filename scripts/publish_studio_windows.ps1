param(
    [switch]$BuildPythonDist,
    [switch]$SkipZip,
    [switch]$SkipBuild,
    [string]$VersionOverride = ''
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($VersionOverride)) {
    $pyprojectMatch = Select-String -Path (Join-Path $repo 'pyproject.toml') -Pattern '^version\s*=\s*"([^"]+)"'
    if (-not $pyprojectMatch) { throw 'Could not determine version from pyproject.toml.' }
    $version = $pyprojectMatch.Matches[0].Groups[1].Value
} else {
    $version = $VersionOverride.Trim()
}

$name = "PolysomStudio-$version-win-x64"
$dist = Join-Path $repo 'dist'
$package = Join-Path $dist $name
$zip = Join-Path $dist "$name.zip"
$publishLocal = Join-Path $repo 'studio\publish'
$publishApp = Join-Path $publishLocal 'app'
$null = New-Item -ItemType Directory -Path $dist -Force
$null = New-Item -ItemType Directory -Path $publishLocal -Force

function Find-SignTool {
    $cmd = Get-Command 'signtool.exe' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    if (Test-Path $kitsRoot) {
        $found = Get-ChildItem -Path $kitsRoot -Recurse -Filter 'signtool.exe' -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

function Invoke-SignFiles {
    param([string[]]$Paths)

    $targets = @($Paths | Where-Object { $_ -and (Test-Path $_) })
    if ($targets.Count -eq 0) { return }

    $hasThumbprint = -not [string]::IsNullOrWhiteSpace($env:SIGNTOOL_CERT_THUMBPRINT)
    $hasPfxBase64 = -not [string]::IsNullOrWhiteSpace($env:SIGNTOOL_CERT_BASE64)
    $hasAzureTrustedSigning = (
        -not [string]::IsNullOrWhiteSpace($env:AZURE_TENANT_ID) -and
        -not [string]::IsNullOrWhiteSpace($env:AZURE_CLIENT_ID) -and
        -not [string]::IsNullOrWhiteSpace($env:AZURE_CLIENT_SECRET) -and
        -not [string]::IsNullOrWhiteSpace($env:TRUSTED_SIGNING_ENDPOINT) -and
        -not [string]::IsNullOrWhiteSpace($env:TRUSTED_SIGNING_ACCOUNT) -and
        -not [string]::IsNullOrWhiteSpace($env:TRUSTED_SIGNING_PROFILE)
    )

    if ($hasThumbprint -or $hasPfxBase64) {
        $signtool = Find-SignTool
        if (-not $signtool) {
            Write-Warning '[Signing] SIGNTOOL_CERT_* configured but signtool.exe was not found in PATH or Windows 10 SDK; skipping signing.'
            return
        }
        $tsUrl = if ([string]::IsNullOrWhiteSpace($env:SIGNTOOL_TIMESTAMP_URL)) { 'http://timestamp.digicert.com' } else { $env:SIGNTOOL_TIMESTAMP_URL }
        $tempPfx = $null
        try {
            if ($hasPfxBase64) {
                $tempPfx = Join-Path $env:TEMP ("polysom_sign_{0}.pfx" -f $PID)
                [System.IO.File]::WriteAllBytes($tempPfx, [System.Convert]::FromBase64String($env:SIGNTOOL_CERT_BASE64))
            }
            foreach ($file in $targets) {
                $signArgs = @('sign', '/fd', 'SHA256', '/tr', $tsUrl, '/td', 'SHA256')
                if ($tempPfx) {
                    $signArgs += @('/f', $tempPfx)
                    if (-not [string]::IsNullOrWhiteSpace($env:SIGNTOOL_CERT_PASSWORD)) {
                        $signArgs += @('/p', $env:SIGNTOOL_CERT_PASSWORD)
                    }
                }
                if ($hasThumbprint) {
                    $signArgs += @('/sha1', $env:SIGNTOOL_CERT_THUMBPRINT)
                }
                $signArgs += $file
                & $signtool @signArgs
                if ($LASTEXITCODE -ne 0) { throw "signtool failed for $file (exit code $LASTEXITCODE)" }
                $sig = Get-AuthenticodeSignature -FilePath $file
                if ($sig.Status -ne 'Valid' -and $sig.Status -ne 'UnknownError') {
                    Write-Warning "[Signing] Authenticode status for $(Split-Path $file -Leaf): $($sig.Status)"
                }
            }
            Write-Host "[Signing] Authenticode-signed $($targets.Count) file(s) via signtool."
        } finally {
            if ($tempPfx -and (Test-Path $tempPfx)) { Remove-Item $tempPfx -Force -ErrorAction SilentlyContinue }
        }
        return
    }

    if ($hasAzureTrustedSigning) {
        $trustedCmd = Get-Command 'Invoke-TrustedSigning' -ErrorAction SilentlyContinue
        if ($trustedCmd) {
            Invoke-TrustedSigning `
                -Endpoint $env:TRUSTED_SIGNING_ENDPOINT `
                -CodeSigningAccountName $env:TRUSTED_SIGNING_ACCOUNT `
                -CertificateProfileName $env:TRUSTED_SIGNING_PROFILE `
                -Files ($targets -join ',') `
                -FileDigest 'SHA256' `
                -TimestampRfc3161 'http://timestamp.acs.microsoft.com'
            Write-Host "[Signing] Signed $($targets.Count) file(s) via TrustedSigningClient."
            return
        }

        $signtool = Find-SignTool
        if ($signtool -and -not [string]::IsNullOrWhiteSpace($env:AZURE_TRUSTED_SIGNING_DLIB) -and (Test-Path $env:AZURE_TRUSTED_SIGNING_DLIB)) {
            $metaJson = Join-Path $env:TEMP ("trusted_signing_meta_{0}.json" -f $PID)
            $metaObj = @{
                Endpoint = $env:TRUSTED_SIGNING_ENDPOINT
                CodeSigningAccountName = $env:TRUSTED_SIGNING_ACCOUNT
                CertificateProfileName = $env:TRUSTED_SIGNING_PROFILE
            } | ConvertTo-Json -Compress
            [System.IO.File]::WriteAllText($metaJson, $metaObj, (New-Object System.Text.UTF8Encoding($false)))
            try {
                foreach ($file in $targets) {
                    & $signtool sign /v /fd SHA256 /tr 'http://timestamp.acs.microsoft.com' /td SHA256 /dlib $env:AZURE_TRUSTED_SIGNING_DLIB /dmdf $metaJson $file
                    if ($LASTEXITCODE -ne 0) { throw "Azure Trusted Signing failed for $file (exit code $LASTEXITCODE)" }
                }
                Write-Host "[Signing] Signed $($targets.Count) file(s) via Azure Trusted Signing dlib."
            } finally {
                Remove-Item $metaJson -Force -ErrorAction SilentlyContinue
            }
            return
        }

        Write-Warning '[Signing] Azure Trusted Signing env vars detected, but neither TrustedSigningClient nor AZURE_TRUSTED_SIGNING_DLIB+signtool is installed; skipping signing.'
        return
    }

    Write-Warning '[Signing] No SIGNTOOL_CERT_THUMBPRINT or Azure Trusted Signing environment variables configured. Binaries remain unsigned (local dev mode).'
}

function Write-Utf8LfFile {
    param([string]$Path, [string]$Text)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $normalized = $Text -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($Path, $normalized, $utf8NoBom)
}

if (-not $SkipBuild) {
    if (Test-Path $publishApp) { Remove-Item $publishApp -Recurse -Force }
    $null = New-Item -ItemType Directory -Path $publishApp -Force

    # Remove stale legacy single-file PolysomStudio.exe in studio/publish/ if present
    $staleRootAppExe = Join-Path $publishLocal 'PolysomStudio.exe'
    if (Test-Path $staleRootAppExe) { Remove-Item $staleRootAppExe -Force }

    foreach ($project in 'PolysomStudio.Windows', 'PolysomStudio.Installer') {
        Write-Host "Publishing structured app/ for $project..."
        dotnet publish (Join-Path $repo "studio\windows\$project\$project.csproj") `
            --configuration Release `
            --runtime win-x64 `
            --self-contained true `
            -p:PublishSingleFile=false `
            --output $publishApp
        if ($LASTEXITCODE -ne 0) { throw "$project publish failed with exit code $LASTEXITCODE" }
    }
}

$lucidBinaries = @(
    (Join-Path $publishApp 'PolysomStudio.exe'),
    (Join-Path $publishApp 'PolysomStudioSetup.exe'),
    (Join-Path $publishApp 'PolysomStudio.dll'),
    (Join-Path $publishApp 'PolysomStudioSetup.dll'),
    (Join-Path $publishApp 'lucid_machines_core.dll')
)
Invoke-SignFiles -Paths $lucidBinaries

if (-not $SkipZip) {
    if (Test-Path $package) { Remove-Item $package -Recurse -Force }
    if (Test-Path $zip) { Remove-Item $zip -Force }
    $null = New-Item -ItemType Directory -Path (Join-Path $package 'scripts'), (Join-Path $package 'studio\windows'), (Join-Path $package 'app') -Force

    Copy-Item (Join-Path $publishApp '*') (Join-Path $package 'app') -Recurse -Force

    foreach ($file in 'pyproject.toml', 'README.md', 'requirements.txt') {
        $srcFile = Join-Path $repo $file
        if (Test-Path $srcFile) { Copy-Item $srcFile $package -Force }
    }
    foreach ($dir in 'docs', 'lucid_machines', 'models', 'studio\src', 'studio\installer') {
        $srcDir = Join-Path $repo $dir
        if (Test-Path $srcDir) { Copy-Item $srcDir (Join-Path $package $dir) -Recurse -Force }
    }
    Copy-Item (Join-Path $repo 'studio\PolysomStudio.bat') (Join-Path $package 'studio') -Force
    $stagedBat = Join-Path $package 'studio\PolysomStudio.bat'
    if (Test-Path $stagedBat) {
        $batText = Get-Content -Path $stagedBat -Raw -Encoding UTF8
        if ($batText -notmatch 'app\\PolysomStudio\.exe') {
            $appCheck = "    if exist `"app\PolysomStudio.exe`" (`r`n        echo [Studio] Launching Native Windows Application ^(app\PolysomStudio.exe^)...`r`n        start `"`" `"app\PolysomStudio.exe`"`r`n        exit /b 0`r`n    )`r`n    if exist `"PolysomStudio.exe`""
            $batText = $batText -replace '    if exist "PolysomStudio\.exe"', $appCheck
            Write-Utf8LfFile -Path $stagedBat -Text ($batText -replace "`n", "`r`n")
        }
    }
    Copy-Item (Join-Path $repo 'studio\windows\polysom.ico') (Join-Path $package 'studio\windows') -Force
    foreach ($script in 'launch_studio.py', 'benchmark_inference.py') {
        Copy-Item (Join-Path $repo "scripts\$script") (Join-Path $package 'scripts') -Force
    }
    Get-ChildItem $package -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem $package -Recurse -File -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force

    Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal
    Remove-Item $package -Recurse -Force
}

# Build standalone PolysomStudioSetup.exe (Inno Setup if available, else single-file fallback)
$standalonePublishSetup = Join-Path $publishLocal 'PolysomStudioSetup.exe'
$standaloneDistSetup = Join-Path $dist 'PolysomStudioSetup.exe'
if (-not $SkipBuild) {
    $isccCmd = Get-Command 'iscc.exe' -ErrorAction SilentlyContinue
    $issPath = Join-Path $repo 'studio\windows\PolysomStudioInstaller.iss'
    if ($isccCmd -and (Test-Path $issPath)) {
        Write-Host "Compiling standalone PolysomStudioSetup.exe via Inno Setup ($($isccCmd.Source))..."
        & $isccCmd.Source "/DMyAppVersion=$version" "/O$publishLocal" "/FPolysomStudioSetup" $issPath
        if ($LASTEXITCODE -ne 0) { throw "iscc failed with exit code $LASTEXITCODE" }
        Copy-Item $standalonePublishSetup $standaloneDistSetup -Force
    } else {
        $tmpSetupDir = Join-Path $env:TEMP ("polysom_setup_single_{0}" -f $PID)
        if (Test-Path $tmpSetupDir) { Remove-Item $tmpSetupDir -Recurse -Force }
        $null = New-Item -ItemType Directory -Path $tmpSetupDir -Force
        try {
            Write-Host 'Publishing standalone single-file PolysomStudioSetup.exe...'
            dotnet publish (Join-Path $repo 'studio\windows\PolysomStudio.Installer\PolysomStudio.Installer.csproj') `
                --configuration Release `
                --runtime win-x64 `
                --self-contained true `
                -p:PublishSingleFile=true `
                -p:IncludeNativeLibrariesForSelfExtract=true `
                --output $tmpSetupDir
            if ($LASTEXITCODE -ne 0) { throw "Standalone PolysomStudio.Installer publish failed with exit code $LASTEXITCODE" }
            $builtSetup = Join-Path $tmpSetupDir 'PolysomStudioSetup.exe'
            Copy-Item $builtSetup $standalonePublishSetup -Force
            Copy-Item $builtSetup $standaloneDistSetup -Force
        } finally {
            Remove-Item $tmpSetupDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    Invoke-SignFiles -Paths @($standalonePublishSetup, $standaloneDistSetup)
}

if ($BuildPythonDist) {
    $studioPy = Join-Path $repo 'studio\studio_env\Scripts\python.exe'
    $pyExe = if (Test-Path $studioPy) { $studioPy } else { 'python' }
    Write-Host "Building Python wheel and sdist with $pyExe..."
    & $pyExe -m build --sdist --wheel --outdir $dist
    if ($LASTEXITCODE -ne 0) { throw "python -m build failed with exit code $LASTEXITCODE" }
}

# Emit deterministic per-artifact .sha256 files and dist/SHA256SUMS.txt (UTF-8 no BOM, LF)
$artifactFiles = Get-ChildItem -Path $dist -File | Where-Object {
    $_.Name -eq "$name.zip" -or
    $_.Name -eq 'PolysomStudioSetup.exe' -or
    $_.Name -like 'lucid_machines-*.whl' -or
    $_.Name -like 'lucid_machines-*.tar.gz'
} | Sort-Object Name

$sumsBuilder = New-Object System.Text.StringBuilder
foreach ($art in $artifactFiles) {
    $hash = (Get-FileHash $art.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $line = "$hash  $($art.Name)`n"
    Write-Utf8LfFile -Path "$($art.FullName).sha256" -Text $line
    $null = $sumsBuilder.Append($line)
}

if ($sumsBuilder.Length -gt 0) {
    Write-Utf8LfFile -Path (Join-Path $dist 'SHA256SUMS.txt') -Text $sumsBuilder.ToString()
}

if (Test-Path $zip) {
    Write-Host "Package ZIP: $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
}
if (Test-Path $standaloneDistSetup) {
    Write-Host "Setup EXE:   $standaloneDistSetup ($([math]::Round((Get-Item $standaloneDistSetup).Length / 1MB, 1)) MB)"
}
Write-Host "App Dir:     $publishApp"
Write-Host "Checksums:   $(Join-Path $dist 'SHA256SUMS.txt')"
