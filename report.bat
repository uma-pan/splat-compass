@echo off
rem Build the summary page and open it in the browser
cd /d "%~dp0"
git pull --ff-only
python -m splat_compass.report
if errorlevel 1 pause
