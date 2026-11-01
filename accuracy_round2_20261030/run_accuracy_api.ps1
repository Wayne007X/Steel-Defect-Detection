$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'E:\yolo_workspace\Scripts\python.exe' -m uvicorn accuracy_api:app --host 127.0.0.1 --port 8001 --workers 1
