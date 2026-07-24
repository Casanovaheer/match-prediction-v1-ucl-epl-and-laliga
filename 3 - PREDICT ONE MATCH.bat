@echo off
cd /d "%~dp0"
title Match Predictions - Single Match
:again
cls
echo.
echo  ============================================================
echo   PREDICT ONE MATCH
echo  ============================================================
echo.
echo   Type the two club names. The home team goes first.
echo   Examples:  Real Madrid  /  Barcelona
echo              Arsenal      /  Man City
echo.

set "HOME_TEAM="
set "AWAY_TEAM="
set /p "HOME_TEAM=  Home team: "
set /p "AWAY_TEAM=  Away team: "

if "%HOME_TEAM%"=="" goto again
if "%AWAY_TEAM%"=="" goto again

echo.
python -m src.predict "%HOME_TEAM%" "%AWAY_TEAM%"

echo.
echo  ------------------------------------------------------------
echo   Press any key to predict another match, or close this window.
echo  ------------------------------------------------------------
pause >nul
goto again
