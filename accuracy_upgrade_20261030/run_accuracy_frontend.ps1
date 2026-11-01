$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\yolo_project'
$env:FASTAPI_BASE_URL = 'http://127.0.0.1:8001'
& 'E:\yolo_workspace\Scripts\python.exe' -m streamlit run .\streamlit_frontend.py --server.address 127.0.0.1 --server.port 8502
