[CmdletBinding()]
param(
    [switch]$SkipTests,
    [string]$BootstrapPython = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildRoot = Join-Path $repoRoot "build\windows"
$distRoot = Join-Path $buildRoot "dist"
$workRoot = Join-Path $buildRoot "pyinstaller"
$venvRoot = Join-Path $buildRoot "venv"
$buildPython = Join-Path $venvRoot "Scripts\python.exe"
$installerRoot = Join-Path $repoRoot "installer"
$specPath = Join-Path $PSScriptRoot "mcr_als.spec"
$innoScript = Join-Path $PSScriptRoot "MCR-ALS.iss"

$isccCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
)
$iscc = $isccCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 6 was not found. Install it with: winget install --id JRSoftware.InnoSetup"
}

New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $installerRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $buildPython)) {
    & $BootstrapPython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the isolated build environment." }
}

$env:PYTHONNOUSERSITE = "1"
& $buildPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Unable to update pip in the build environment." }
& $buildPython -m pip install --disable-pip-version-check `
    "pyinstaller==6.21.0" `
    "numpy==1.26.4" `
    "scipy==1.15.2" `
    "matplotlib==3.9.2" `
    "openpyxl==3.1.5" `
    "pytest>=8,<9"
if ($LASTEXITCODE -ne 0) { throw "Unable to install the pinned build dependencies." }
& $buildPython -m pip install --disable-pip-version-check --no-deps --editable $repoRoot
if ($LASTEXITCODE -ne 0) { throw "Unable to install MCR-ALS in the build environment." }

if (-not $SkipTests) {
    Push-Location $repoRoot
    try {
        & $buildPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "The test suite failed." }
    }
    finally {
        Pop-Location
    }
}

Push-Location $repoRoot
try {
    & $buildPython -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $distRoot `
        --workpath $workRoot `
        $specPath
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

    $applicationDir = Join-Path $distRoot "MCR-ALS"
    & $iscc "/DSourceDir=$applicationDir" "/DOutputDir=$installerRoot" $innoScript
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }
}
finally {
    Pop-Location
}

$installer = Join-Path $installerRoot "MCR-ALS-Setup-0.1.0-Windows-x64.exe"
if (-not (Test-Path -LiteralPath $installer)) {
    throw "The expected installer was not produced: $installer"
}

$hash = Get-FileHash -LiteralPath $installer -Algorithm SHA256
Write-Output "Installer: $installer"
Write-Output "Size: $((Get-Item -LiteralPath $installer).Length) bytes"
Write-Output "SHA256: $($hash.Hash)"
