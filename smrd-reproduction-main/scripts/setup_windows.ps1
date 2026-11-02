# Create the project virtual environment and install dependencies.
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Checking Python 3.12"
$python = "py"
& $python -3.12 -c "import sys; print(sys.version)" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 not found. Install it with: winget install Python.Python.3.12"
}

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating .venv"
    & $python -3.12 -m venv .venv
}

$vpy = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "==> Upgrading pip"
& $vpy -m pip install --upgrade pip wheel setuptools

Write-Host "==> Installing PyTorch (CUDA 12.8 wheel)"
& $vpy -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

Write-Host "==> Installing remaining requirements"
& $vpy -m pip install -r requirements.txt

Write-Host "==> Downloading NLTK corpora"
& $vpy -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords','wordnet','omw-1.4','punkt')]"

Write-Host "==> Verifying CUDA"
& $vpy -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "==> Running tests"
& $vpy -m pytest -q

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
