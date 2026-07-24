@echo off
cd /d "%~dp0"
title Match Predictions - Show Results
echo.
echo  ============================================================
echo   SHOWING RESULTS
echo  ============================================================
echo.

if not exist "docs\index.html" (
    echo   No results yet. Run "2 - UPDATE EVERYTHING.bat" first.
    echo.
    pause
    exit /b 1
)

echo   Opening results in your browser...
start "" "docs\index.html"
echo.
echo   Done. You can close this window.
echo.
timeout /t 4 >nul
