$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'E:\yolo_workspace\Scripts\python.exe' -m uvicorn rebuilt_api:app --host 127.0.0.1 --port 8002 --workers 1
