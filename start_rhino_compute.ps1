
# Start Rhino Compute server from local project folder
$rhinoComputePath = Join-Path $PSScriptRoot "rhino.compute\rhino.compute\rhino.compute.exe"

if (Test-Path $rhinoComputePath) {
    Write-Host "Starting Rhino Compute from local folder: $rhinoComputePath"
    Write-Host "Port: 81"
    & $rhinoComputePath --port 81
} else {
    Write-Error "Rhino Compute executable not found at: $rhinoComputePath"
}
