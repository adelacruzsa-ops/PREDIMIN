@echo off
REM PREDIMIN - Abre la aplicacion web en el navegador. Para cerrarla, cierre esta ventana.
cd /d "%~dp0"
if not exist "models\predimin_pm10_ugm3.joblib" (
    echo Primero ejecute 2_EJECUTAR_TODO.bat para entrenar los modelos.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run src\app.py
pause
