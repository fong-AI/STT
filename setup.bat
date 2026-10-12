@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import ctranslate2, sys; sys.exit(0 if ctranslate2.get_cuda_device_count() else 1)"
if errorlevel 1 goto selfcheck
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 goto failed
:selfcheck
".venv\Scripts\python.exe" meeting_stt.py --self-check
if errorlevel 1 goto failed
echo Setup complete. Open run.bat to begin.
pause
exit /b 0
:failed
echo Setup failed. See the error above.
pause
exit /b 1
