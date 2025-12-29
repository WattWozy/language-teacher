
Write-Host "Starting Local Voice Assistant..." -ForegroundColor Cyan

# 1. Backend Setup & Run
$BackendDir = Join-Path $PSScriptRoot "backend"
if (Test-Path $BackendDir) {
    Write-Host "Checking Backend configuration..." -ForegroundColor Yellow
    Push-Location $BackendDir
    
    # Check for venv
    if (-not (Test-Path "venv")) {
        Write-Host "Creating Python Virtual Environment (this may take a minute)..." -ForegroundColor Cyan
        python -m venv venv
        
        Write-Host "Installing Dependencies..." -ForegroundColor Cyan
        .\venv\Scripts\python -m pip install --upgrade pip
        .\venv\Scripts\pip install -r requirements.txt
        
        # Check for NVIDIA/CUDA support if on Windows
        try {
            if ((Get-CimInstance Win32_VideoController).Name -match "NVIDIA") {
                 Write-Host "NVIDIA GPU detected. Installing CUDA libraries..." -ForegroundColor Cyan
                 .\venv\Scripts\pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
            }
        } catch {}
    }
    
    Write-Host "Launching Backend Server in new window..." -ForegroundColor Green
    # We use Start-Process to open a new window so this script doesn't block
    # Using python -m uvicorn is more robust than relying on the shim being in PATH immediately
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { Write-Host 'Backend Server'; cd '$BackendDir'; . .\venv\Scripts\Activate.ps1; .\venv\Scripts\python -m uvicorn server:app --host 0.0.0.0 --port 8000 }"
    
    Pop-Location
} else {
    Write-Error "Backend directory not found!"
}

# 2. Frontend Run
$FrontendDir = Join-Path $PSScriptRoot "frontend"
if (Test-Path $FrontendDir) {
    Write-Host "Launching Frontend in new window..." -ForegroundColor Green
    Push-Location $FrontendDir
    
    # Optional: Check node_modules (simple check)
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing Frontend Dependencies..." -ForegroundColor Cyan
        npm install
    }
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& { Write-Host 'Frontend Server'; cd '$FrontendDir'; npm run dev }"
    Pop-Location
} else {
    Write-Error "Frontend directory not found!"
}

Write-Host "---------------------------------------------------" -ForegroundColor White
Write-Host "App is starting!" -ForegroundColor Green
Write-Host "Access the app at: http://localhost:3000" -ForegroundColor Cyan
Write-Host "---------------------------------------------------" -ForegroundColor White
Start-Sleep -Seconds 3
