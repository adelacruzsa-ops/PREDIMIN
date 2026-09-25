@echo off
REM Igual que 2_EJECUTAR_TODO.bat pero sin optimizacion bayesiana (1-3 min). Para pruebas.
cd /d "%~dp0"
call 2_EJECUTAR_TODO.bat --trials 0
