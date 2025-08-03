# Start Rhino Compute server on port 6501 - Auto-detect user and Hops version

# Get current username
$username = $env:USERNAME

# Find the latest Hops version
$hopsBasePath = "C:\Users\$username\AppData\Roaming\McNeel\Rhinoceros\packages\8.0\Hops"

if (Test-Path $hopsBasePath) {
    $latestHopsVersion = Get-ChildItem -Path $hopsBasePath -Directory | 
                        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
                        Sort-Object { [version]$_.Name } -Descending |
                        Select-Object -First 1 -ExpandProperty Name
    
    if ($latestHopsVersion) {
        $rhinoComputePath = "$hopsBasePath\$latestHopsVersion\rhino.compute\rhino.compute.exe"
        
        if (Test-Path $rhinoComputePath) {
            Write-Host "Starting Rhino Compute with:"
            Write-Host "  User: $username"
            Write-Host "  Hops Version: $latestHopsVersion"
            Write-Host "  Path: $rhinoComputePath"
            Write-Host "  Port: 6501"
            
            & $rhinoComputePath --port 6501
        } else {
            Write-Error "Rhino Compute executable not found at: $rhinoComputePath"
        }
    } else {
        Write-Error "No valid Hops version found in: $hopsBasePath"
    }
} else {
    Write-Error "Hops installation not found at: $hopsBasePath"
}
