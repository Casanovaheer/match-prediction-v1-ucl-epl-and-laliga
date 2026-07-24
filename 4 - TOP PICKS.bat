@echo off
cd /d "%~dp0"
title Match Predictions - Top Picks
:again
cls
echo.
echo  ============================================================
echo   TOP PICKS FOR ONE MATCH
echo   Ranked selections with a confidence rating on each.
echo  ============================================================
echo.
echo   Home team first, then away team.
echo.
echo   Examples:   Real Madrid   /  Barcelona      (full model)
echo               Arsenal       /  Man City       (full model)
echo               Degerfors     /  Djurgarden     (Elo estimate)
echo.

set "HOME_TEAM="
set "AWAY_TEAM="
set /p "HOME_TEAM=  Home team: "
set /p "AWAY_TEAM=  Away team: "

if "%HOME_TEAM%"=="" goto again
if "%AWAY_TEAM%"=="" goto again

echo.
python -m src.toppick "%HOME_TEAM%" "%AWAY_TEAM%"

echo.
echo  ------------------------------------------------------------
echo   Press any key for another match, or close this window.
echo  ------------------------------------------------------------
pause >nul
goto again
