@echo off
title Reaching Lab
set GLOG_minloglevel=2
set TF_CPP_MIN_LOG_LEVEL=3
cd /d "%~dp0"
set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=..\venv\Scripts\python.exe
"%PY%" lab_app.py
if errorlevel 1 pause
