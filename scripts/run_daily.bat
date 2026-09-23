@echo off
rem KeibaAI daily run. Register this file in Windows Task Scheduler (race days, 8:30).
rem Arguments are passed to main.py as-is. Example: run_daily.bat --paper
cd /d "%~dp0.."
".venv\Scripts\python.exe" main.py %*
exit /b %ERRORLEVEL%
