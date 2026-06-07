$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    streamlit run .\src\app.py
} finally {
    Pop-Location
}
