@echo off
rem CAMERA: check with  python -m splat_compass.live --list
set CAMERA=0

cd /d "%~dp0"
git pull --ff-only
python -m splat_compass.live --camera %CAMERA%
if errorlevel 1 pause
