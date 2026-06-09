$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    python -m src.train --mode demo --epochs 5 --days 60 --lookback 48 --future-hours 336 --model transformer
} finally {
    Pop-Location
}
