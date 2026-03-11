Write-Host "Starting FastAPI server..."

Start-Process powershell -ArgumentList "cd C:\Users\kpolice\rag-debug-system; .\venv\Scripts\python.exe -m uvicorn app.api.main:app --reload"

Start-Sleep -Seconds 2

Write-Host "Starting Streamlit UI..."

Start-Process powershell -ArgumentList "cd C:\Users\kpolice\rag-debug-system; .\venv\Scripts\python.exe -m streamlit run ui/app.py"

Write-Host "Application started!"