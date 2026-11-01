$ErrorActionPreference = 'Stop'
$splitTarget = Join-Path $PSScriptRoot 'tools\split_deps'
& 'E:\yolo_workspace\Scripts\python.exe' -m pip install --disable-pip-version-check --no-deps --target $splitTarget -r (Join-Path $PSScriptRoot 'requirements-split.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install isolated split dependencies.' }
