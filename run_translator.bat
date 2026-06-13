@echo off
title SignSpeak Translator
echo ==============================================
echo       SignSpeak Sign Language Translator
echo ==============================================
echo.
echo [1/2] Navigating to project folder...
cd /d "%~dp0"
echo [2/2] Starting Flask server on Python 3.11...
echo.
echo -- once started, open http://127.0.0.1:5000 in your browser --
echo -- press Ctrl+C in this window to stop --
echo.
py -3.11 app.py
pause
