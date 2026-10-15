# Share the app from this PC on a public link.
#
#   Double-click share.bat, or:  powershell -ExecutionPolicy Bypass -File share.ps1
#
# 1. builds the website once if frontend\dist is missing,
# 2. starts the backend (it also serves the website) if it is not running,
# 3. opens a free Cloudflare quick tunnel and prints the https link.
#
# Keep this window open: closing it ends the link. The PC must stay on and
# awake. The link changes every time this script is started.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
$port = 8000

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

if (Test-Backend) {
    Write-Host "Backend already running on port $port."
} else {
    Write-Host "Starting the backend (minimised window titled 'python')..."
    Start-Process -FilePath $python -ArgumentList "-m", "app" -WorkingDirectory (Join-Path $root "backend") -WindowStyle Minimized
    for ($i = 0; $i -lt 90 -and -not (Test-Backend); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Backend)) { throw "The backend did not start. Run 'python -m app' in backend\ to see the error." }
}

$info = Invoke-RestMethod "http://127.0.0.1:$port/api/info"
Write-Host "Mode: $($info.mode) - $($info.message)"

$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) { $cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe" }
if (-not (Test-Path $cloudflared)) { throw "cloudflared is not installed: winget install --id Cloudflare.cloudflared" }

Write-Host ""
Write-Host "Opening the public link. Look for the line with https://....trycloudflare.com below."
Write-Host "Anyone with that link can use the app. Do not upload identifiable patient data."
Write-Host ""
& $cloudflared tunnel --no-autoupdate --url "http://localhost:$port"
