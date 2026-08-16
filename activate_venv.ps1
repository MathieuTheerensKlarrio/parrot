## activate_venv.ps1
# Launches a new PowerShell with the repository `venv` activated.
# Usage: `powershell -ExecutionPolicy Bypass -File .\activate_venv.ps1`

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPath = Join-Path $repoRoot 'venv'

if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtualenv not found at $venvPath." -ForegroundColor Yellow
    Write-Host "Create one with: python -m venv venv"
    exit 1
}

$activatePs1 = Join-Path $venvPath 'Scripts\Activate.ps1'
if (Test-Path $activatePs1) {
    Write-Host "Opening PowerShell with venv activated..." -ForegroundColor Green
    Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass',"-Command","& { . '$activatePs1' }"
    exit 0
}

$activateBat = Join-Path $venvPath 'Scripts\activate.bat'
if (Test-Path $activateBat) {
    Write-Host "Opening cmd with venv activated (activate.bat found)..." -ForegroundColor Green
    Start-Process cmd -ArgumentList '/k',"%CD%\\venv\\Scripts\\activate.bat"
    exit 0
}

Write-Host "No activation script found in $venvPath\\Scripts" -ForegroundColor Red
exit 2
