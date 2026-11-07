@echo off
rem Open the app without reading the camera (summary only)
cd /d "%~dp0"
git pull --ff-only
python -m pip install -q -r requirements.txt
python -m splat_compass.app
if errorlevel 1 pause
