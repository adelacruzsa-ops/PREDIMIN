@echo off
REM PREDIMIN - Procesa los datos, entrena los modelos y genera el reporte.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo Primero ejecute 1_INSTALAR.bat
    pause
    exit /b 1
)
echo === 1/5 Importando y depurando el Excel ===
"%PY%" src\importar_excel.py || goto error
echo === 2/5 Analisis exploratorio ===
"%PY%" src\analisis_exploratorio.py || goto error
echo === 3/5 Entrenando modelos (15-40 min; para prueba rapida use --trials 0) ===
"%PY%" src\entrenamiento.py %* || goto error
echo === 4/5 Red neuronal: analisis detallado (3-8 min) ===
"%PY%" src\red_neuronal.py || goto error
echo === 5/5 Reporte de resultados para la tesis ===
"%PY%" src\reporte_tesis.py || goto error
echo.
echo === LISTO. Resultados en la carpeta outputs. Abra la app con 3_ABRIR_APP.bat ===
pause
exit /b 0
:error
echo.
echo *** Ocurrio un error. Tome una captura de esta ventana. ***
pause
exit /b 1
