@echo off
rem 最新版に更新してから、キャプボ（OBS仮想カメラ）の読み取りを始める
rem カメラ番号は python -m splat_compass.live --list で調べて、下の数字を書き換える
set CAMERA=0

cd /d %~dp0
git pull --ff-only
python -m splat_compass.live --camera %CAMERA%
if errorlevel 1 pause
