@echo off
rem Download stat.ink public battle data (last 30 days) and summarize it
cd /d "%~dp0"
git pull --ff-only
python -m splat_compass.statink
pause
