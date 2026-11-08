@echo off
chcp 65001 >nul
rem 配る版（exe）を作る。できたものは dist\splat-compass フォルダ（中の splat-compass.exe を開く）
cd /d "%~dp0"
python -m pip install -q -r requirements.txt pyinstaller
python -m splat_compass.make_defaults
if errorlevel 1 goto fail
python -m PyInstaller --noconfirm --clean --name splat-compass --windowed ^
  --add-data "splat_compass\templates;splat_compass\templates" ^
  --add-data "splat_compass\defaults;splat_compass\defaults" ^
  --collect-all webview ^
  launcher.py
if errorlevel 1 goto fail
copy /y "はじめに.txt" "dist\splat-compass\はじめに.txt" >nul
for /f %%v in ('python -c "import splat_compass; print(splat_compass.__version__)"') do set VER=%%v
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\splat-compass -DestinationPath dist\splat-compass-%VER%.zip"
echo.
echo できました: dist\splat-compass-%VER%.zip （これを配ります）
pause
exit /b 0
:fail
echo 失敗しました。上のメッセージを確かめてください。
pause
exit /b 1
