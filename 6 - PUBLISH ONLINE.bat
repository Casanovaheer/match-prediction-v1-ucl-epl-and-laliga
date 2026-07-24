@echo off
cd /d "%~dp0"
title Publish Predictions Online
cls
echo.
echo  ============================================================
echo   PUBLISH THE ONLINE MATCH PAGE
echo   Rebuilds the numbers and pushes them to GitHub so your
echo   phone / web page shows the latest ratings.
echo  ============================================================
echo.
echo   Rebuilding predictions...
python -m src.report
if errorlevel 1 goto fail
echo.
echo   Uploading to GitHub...
git add docs
git commit -m "Update online predictions" 2>nul
git push
if errorlevel 1 goto fail
echo.
echo  ------------------------------------------------------------
echo   DONE. Your page is live at:
echo   https://casanovaheer.github.io/match-prediction-v1-ucl-epl-and-laliga/match.html
echo  ------------------------------------------------------------
echo.
echo   Open that link on any phone or computer, type two teams.
pause >nul
exit /b 0

:fail
echo.
echo  ------------------------------------------------------------
echo   Something went wrong. Check the messages above.
echo   (Usually: not logged in to GitHub, or no internet.)
echo  ------------------------------------------------------------
pause >nul
exit /b 1
