@echo off
echo Starting Transcriber...
python start.py
if errorlevel 1 (
    echo.
    echo [error] Something went wrong. See message above.
    echo If Python is not installed: https://www.python.org/downloads/
    pause
)
