$directory = ".venv"
$configFile = "config.json"
$exampleConfigFile = "example.config.json"

if (-Not (Test-Path $directory)) {
    Write-Host "$directory does not exist."
    Write-Host "Creating $directory."
    python -m venv $directory
    Write-Host "Installing dependencies..."
    $env:PATH = "$PWD\$directory\Scripts;$env:PATH"
    & "$PWD\$directory\Scripts\pip.exe" install -r requirements.txt --no-cache-dir
    & "$PWD\$directory\Scripts\pip.exe" install -r requirements.txt --no-cache-dir
}

if (-Not (Test-Path $configFile)) {
    Write-Host "$configFile doesn't exist"
    Write-Host "Cloning it from $exampleConfigFile"
    Copy-Item $exampleConfigFile $configFile
}

$env:PATH = "$PWD\$directory\Scripts;$env:PATH"
& "$PWD\$directory\Scripts\python.exe" main.py
