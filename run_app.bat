@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -X utf8 -m streamlit run "%~dp0app.py" %*
) else (
    python -X utf8 -m streamlit run "%~dp0app.py" %*
)

if errorlevel 1 (
    echo.
    echo Khong the chay ung dung. Xem loi o tren.
    pause
)
endlocal
