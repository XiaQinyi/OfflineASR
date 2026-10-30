@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python virtual environment not found: %PYTHON%
  echo Read README first and create .venv.
  pause
  exit /b 1
)
"%PYTHON%" -m pip install pyinstaller
if errorlevel 1 goto :fail
"%PYTHON%" -m PyInstaller --noconfirm --clean --onedir --windowed --name OfflineASR --collect-all wenet --collect-all whisper --collect-all imageio_ffmpeg --collect-all sentencepiece app.py
if errorlevel 1 goto :fail
if not exist dist\OfflineASR\model mkdir dist\OfflineASR\model
xcopy /E /I /Y model dist\OfflineASR\model >nul
if errorlevel 1 goto :fail
echo.
echo EXE path: dist\OfflineASR\OfflineASR.exe
echo Test on another offline Windows computer before submission.
pause
exit /b 0
:fail
echo Build failed. Please review the error above.
pause
exit /b 1
