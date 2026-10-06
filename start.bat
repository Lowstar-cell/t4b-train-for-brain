@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo Train4brain 2.0 - Interface auf http://127.0.0.1:7865
python interface.py
pause
