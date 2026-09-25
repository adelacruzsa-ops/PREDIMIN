@echo off
REM PREDIMIN - Instalacion (solo la primera vez). Doble clic para ejecutar.
cd /d "%~dp0"
echo === Creando entorno virtual .venv ===
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: no se encontro Python. Instale Python 3.12 marcando "Add Python to PATH".
    pause
    exit /b 1
)
echo === Instalando librerias (puede tardar varios minutos) ===
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo === LISTO. Ahora ejecute 2_EJECUTAR_TODO.bat ===
pause
