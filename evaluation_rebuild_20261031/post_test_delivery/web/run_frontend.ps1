$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:FASTAPI_BASE_URL = 'http://127.0.0.1:8002'
& 'E:\yolo_workspace\Scripts\python.exe' -m streamlit run .\streamlit_rebuilt.py --server.address 127.0.0.1 --server.port 8503
