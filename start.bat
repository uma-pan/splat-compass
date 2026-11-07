@echo off
rem Camera number: write it in camera.txt (check with: python -m splat_compass.live --list)
set CAMERA=0
cd /d "%~dp0"
if exist camera.txt set /p CAMERA=<camera.txt

git pull --ff-only
python -m pip install -q -r requirements.txt
python -m splat_compass.app --camera %CAMERA%
if errorlevel 1 pause
