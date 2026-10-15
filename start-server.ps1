# Run the app on its permanent public address:
#
#   https://glioma.tail7d9208.ts.net
#
# Double-click start-server.bat, or:  powershell -ExecutionPolicy Bypass -File start-server.ps1
#
# Tailscale Funnel is already configured (it is saved by Tailscale and comes
# back after a reboot); this script makes sure it is on and starts the backend,
# which serves the website and runs the model. Keep the window open: closing it
# stops the app, and the address shows an error until it is started again.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
$tailscale = "C:\Program Files\Tailscale\tailscale.exe"
$port = 8000
$address = "https://glioma.tail7d9208.ts.net"

function Test-Backend {
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:$port/api/info" | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    Write-Host "Building the website (first run only)..."
    Push-Location (Join-Path $root "frontend")
    npm run build
    Pop-Location
}

if (Test-Path $tailscale) {
    # Idempotent: re-applies the saved Funnel setting if it was switched off.
    & $tailscale funnel --bg $port | Out-Null
} else {
    Write-Warning "Tailscale is not installed; the app will only be reachable on this PC."
}

if (Test-Backend) {
    Write-Host "The app is already running."
    Write-Host "Public address: $address"
    exit 0
}

Write-Host "Starting the app. Public address: $address"
Write-Host "Keep this window open. Do not upload identifiable patient data: anyone with the address can use the app."
Write-Host ""
Set-Location (Join-Path $root "backend")
$env:GLIOMA_HOST = "127.0.0.1"
$env:GLIOMA_PORT = "$port"
& $python -m app
