python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Write-Host "Setup complete. Virtualenv active and dependencies installed."
