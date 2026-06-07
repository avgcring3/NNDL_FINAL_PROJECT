$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$BasePort = 8503
if ($env:RIDEFLOW_PORT) {
    $BasePort = [int]$env:RIDEFLOW_PORT
}

function Test-LocalPortOpen {
    param([int]$Port)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(250, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Get-FreeDashboardPort {
    param([int]$StartPort)

    for ($candidate = $StartPort; $candidate -lt ($StartPort + 30); $candidate++) {
        if (-not (Test-LocalPortOpen -Port $candidate)) {
            return $candidate
        }
        Write-Host "Port $candidate is busy; trying next."
    }
    throw "No free dashboard port found from $StartPort to $($StartPort + 29)."
}

Push-Location $ProjectRoot
try {
    Write-Host "RideFlow NN launcher"

    $pythonVersion = python --version
    Write-Host "Python: $pythonVersion"

    $checkCode = "import importlib.util,sys; mods=['pandas','numpy','torch','streamlit','matplotlib','sklearn','pyarrow','requests','folium','streamlit_folium']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing)); sys.exit(1 if missing else 0)"
    python -c $checkCode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing missing dependencies..."
        python -m pip install -r requirements.txt
    }

    $metrics = Join-Path $ProjectRoot "artifacts\metrics.json"
    $predictions = Join-Path $ProjectRoot "artifacts\predictions.csv"
    $future = Join-Path $ProjectRoot "artifacts\future_forecast.csv"
    $needsTraining = $false
    if ((-not (Test-Path -LiteralPath $metrics)) -or (-not (Test-Path -LiteralPath $predictions)) -or (-not (Test-Path -LiteralPath $future))) {
        $needsTraining = $true
    } else {
        $futureHours = python -c "import json; from pathlib import Path; p=Path(r'$metrics'); print(json.loads(p.read_text()).get('metadata',{}).get('future_hours',0))"
        $futureFresh = python -c "import pandas as pd; from pathlib import Path; p=Path(r'$future'); d=pd.read_csv(p, usecols=['timestamp']); end=pd.to_datetime(d['timestamp']).max(); now=pd.Timestamp.now(tz='Europe/Moscow').tz_localize(None); print(1 if end >= now + pd.Timedelta(days=13) else 0)"
        if (([int]$futureHours -lt 336) -or ([int]$futureFresh -ne 1)) {
            $needsTraining = $true
        }
    }

    if ($needsTraining) {
        Write-Host "Training demo model and generating current 14-day future forecast..."
        python -m src.train --mode demo --city moscow --epochs 5 --days 60 --lookback 48 --future-hours 336 --model transformer
    } else {
        Write-Host "Existing artifacts found. Skipping training."
    }

    $pricingCheck = "import sys; sys.path.insert(0, r'$ProjectRoot\src'); from ride_pricing import automatic_traffic_factor; print('pricing_import_ok')"
    python -c $pricingCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Pricing module import check failed."
    }

    $Port = Get-FreeDashboardPort -StartPort $BasePort
    $Url = "http://127.0.0.1:$Port"
    Set-Content -LiteralPath (Join-Path $ProjectRoot "dashboard_url.txt") -Value $Url -Encoding UTF8

    Write-Host "Starting fresh dashboard on $Url ..."
    $arguments = @("-m", "streamlit", "run", (Join-Path $ProjectRoot "src\app.py"), "--server.port", "$Port", "--server.headless", "true", "--server.fileWatcherType", "poll")
    Start-Process -FilePath python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden

    $ready = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Seconds 1
        try {
            $status = (Invoke-WebRequest -Uri "$Url/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode
            if ($status -eq 200) {
                $ready = $true
                break
            }
        } catch {
        }
    }
    if (-not $ready) {
        throw "Dashboard did not become ready on $Url"
    }

    Start-Process $Url
    Write-Host "Opened $Url"
} finally {
    Pop-Location
}
