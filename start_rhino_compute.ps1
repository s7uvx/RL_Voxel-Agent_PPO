
# Start Rhino Compute Coordinator
# The coordinator runs on port 81 and automatically manages child processes
# All training environments connect to the same coordinator port
param(
    [int]$Port = 81
)

$rhinoComputePath = Join-Path $PSScriptRoot "rhino.compute\rhino.compute\rhino.compute.exe"

if (Test-Path $rhinoComputePath) {
    Write-Host "="*60
    Write-Host "Starting Rhino.Compute Coordinator"
    Write-Host "="*60
    Write-Host "Coordinator Port: $Port"
    Write-Host "Architecture: Coordinator will spawn child processes as needed"
    Write-Host "All training environments will connect to port $Port"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the coordinator and all child processes"
    Write-Host "="*60
    Write-Host ""
    
    & $rhinoComputePath --port $Port
} else {
    Write-Error "Rhino Compute executable not found at: $rhinoComputePath"
    Write-Host ""
    Write-Host "Make sure the rhino.compute folder structure exists:"
    Write-Host "  rhino.compute\"
    Write-Host "    rhino.compute\"
    Write-Host "      rhino.compute.exe"
}
