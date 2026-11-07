@echo off
rem Cut weapon icons from past recordings, then open the naming window
cd /d "%~dp0"
git pull --ff-only
python -m splat_compass.weapons --extract
python -m splat_compass.weapons --label
python -m splat_compass.report
if errorlevel 1 pause
