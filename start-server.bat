@echo off
rem Starts the app on its permanent address https://glioma.tail7d9208.ts.net
rem Keep this window open while people use it.
title Glioma segmentation app - https://glioma.tail7d9208.ts.net
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-server.ps1"
pause
