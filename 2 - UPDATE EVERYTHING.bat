@echo off
cd /d "%~dp0"
title Match Predictions - Update Everything
echo.
echo  ============================================================
echo   UPDATING EVERYTHING
echo   Downloads the latest results, rechecks the model, and
echo   rebuilds all predictions. Takes about 5 minutes.
echo  ============================================================
echo.

echo  [1/5] Downloading latest results...
python -m src.collect
if errorlevel 1 goto failed

echo.
echo  [2/5] Cleaning and validating...
python -m src.clean
if errorlevel 1 goto badd_ata

echo.
echo  [3/5] Checking the model still behaves...
python -m tests.verify_model >nul
if errorlevel 1 goto badmodel

echo.
echo  [4/5] Simulating Champions League...
python -m src.ucl --sims 2000

echo.
echo  [5/5] Building the results page...
python -m src.report
if errorlevel 1 goto failed

echo.
echo  ============================================================
echo   DONE. Opening results...
echo  ============================================================
start "" "docs\index.html"
timeout /t 5 >nul
exit /b 0

:badd_ata
echo.
echo  ============================================================
echo   STOPPED - the downloaded data failed its checks.
echo   Nothing was published. The messages above say what broke.
echo  ============================================================
pause
exit /b 1

:badmodel
echo.
echo  ============================================================
echo   STOPPED - the model failed its own tests.
echo   Nothing was published. Run this to see why:
echo      python -m tests.verify_model
echo  ============================================================
pause
exit /b 1

:failed
echo.
echo  Something went wrong. The messages above explain what.
pause
exit /b 1
