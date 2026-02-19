@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYTHON=%~dp0labelStudioVenv\Scripts\python.exe"

:: --- LIMPEZA DE MEMÓRIA ---
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR="

set "ENV_FILE=.env"

:init_config
if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"

:: Carrega variáveis do arquivo .env 
for /f "usebackq tokens=*" %%i in ("%ENV_FILE%") do (
    set "line=%%i"
    if "!line:~0,1!" neq "#" if not "!line!"=="" (
        for /f "tokens=1,2 delims==" %%a in ("!line!") do set "%%a=%%b"
    )
)

:: --- CONFIGURAÇÃO INICIAL ---
if "!YOLO_PLATE_MODEL_PATH!"=="" (
    set "YOLO_PLATE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\models\best.pt"
)
if "!YOLO_VEHICLE_MODEL_PATH!"=="" (
    set "YOLO_VEHICLE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11\models\yolo11x.pt"
)

if "%DEVICE%"=="" (
    echo.
    set /p "IN_DEV=[CONFIGURACAO] Dispositivo (cpu / cuda): " 
    if not "!IN_DEV!"=="" (
        set "DEVICE=!IN_DEV!"
        echo DEVICE=!DEVICE!>> "%ENV_FILE%"
    ) else ( goto init_config )
)

if "%LABEL_STUDIO_URL%"=="" (
    echo.
    set /p "IN_URL=[CONFIGURACAO] URL do Label Studio (Enter para padrao: http://localhost:8080): "
    if "!IN_URL!"=="" ( set "LABEL_STUDIO_URL=http://localhost:8080" ) else ( set "LABEL_STUDIO_URL=!IN_URL!" )
    echo LABEL_STUDIO_URL=!LABEL_STUDIO_URL!>> "%ENV_FILE%"
)

if "%LABEL_STUDIO_ML_BACKEND_URL%"=="" (
    echo.
    set /p "IN_ML_URL=[CONFIGURACAO] URL do Backend ML (Enter para padrao: http://localhost:9090): "
    if "!IN_ML_URL!"=="" ( set "LABEL_STUDIO_ML_BACKEND_URL=http://localhost:9090" ) else ( set "LABEL_STUDIO_ML_BACKEND_URL=!IN_ML_URL!" )
    echo LABEL_STUDIO_ML_BACKEND_URL=!LABEL_STUDIO_ML_BACKEND_URL!>> "%ENV_FILE%"
)

:menu
cls
echo ==========================================================
echo         Label Studio Auto-Labeling Tool (Windows)
echo ==========================================================
echo CONFIG ATUAL: %DEVICE%  
echo CONFIG FRONTEND: %LABEL_STUDIO_URL%
echo CONFIG BACKEND : %LABEL_STUDIO_ML_BACKEND_URL%
echo MODEL PADRAO: !SELECTED_BACKEND!
echo ========================================
echo 1. Iniciar servidores
echo 2. Auto-labeling interativo
echo 3. Parar servidores
echo 4. Resetar configuracoes (.env)
echo 5. Sair
echo ========================================
set "CHOICE="
set /p "CHOICE=Escolha uma opcao: "

if "!CHOICE!"=="1" goto start_servers
if "!CHOICE!"=="2" goto auto_label
if "!CHOICE!"=="3" goto stop_servers
if "!CHOICE!"=="4" goto reset_env
if "!CHOICE!"=="5" exit
goto menu

:start_servers
echo.
echo [1] Iniciar Tudo (Frontend + Backend)
echo [2] Apenas Frontend (Label Studio)
echo [3] Apenas Backend (Label Studio ML)
set /p "OP_START=Escolha uma opcao: "

:: Define a pasta do backend
set "B_DIR=segment_anything_2_image"
if /i "!SELECTED_BACKEND!"=="YOLO" set "B_DIR=yolov11-plate"

:: Validação: verifica se a pasta do backend existe antes de tentar abrir
if "%OP_START%"=="1" if not exist ".\label-studio-ml-backend\label_studio_ml\examples\!B_DIR!" (
    echo [ERRO] Pasta .\label-studio-ml-backend\label_studio_ml\examples\!B_DIR! nao encontrada.
    pause
    goto menu
)

if "%OP_START%"=="1" (
    echo Iniciando Frontend e Backend...
    :: Inicia o Frontend (Minimizado)
    start "LS-Frontend" /min cmd /c "call .\labelStudioVenv\Scripts\activate.bat && label-studio start"
    
    timeout /t 2 /nobreak >nul
    
    :: Inicia o Backend (Minimizado / Usei /k para voce ver erros se ele fechar)
    start "LS-Backend" /min cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!B_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && label-studio-ml start ."
    echo Servidores iniciados. 
    echo Servidores configurados para: !LABEL_STUDIO_URL! e !LABEL_STUDIO_ML_BACKEND_URL!
)

if "%OP_START%"=="2" (
    start "LS-Frontend" /min cmd /c "call .\labelStudioVenv\Scripts\activate.bat && label-studio start"
    echo Frontend iniciado.
    echo Servidor configurado para: !LABEL_STUDIO_URL!
)

if "%OP_START%"=="3" (
    start "LS-Backend" /min cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!B_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && label-studio-ml start ."
    echo Backend iniciado.
    echo Servidor configurado para: !LABEL_STUDIO_ML_BACKEND_URL!
)

echo.
pause
goto menu

:select_backend_folder
set "FINAL_DIR=!SELECTED_BACKEND!"
if "!SELECTED_BACKEND!"=="YOLO" (
    echo.
    echo Qual modelo YOLO deseja iniciar no backend?
    echo [1] Veiculos (yolov11)
    echo [2] Placas (yolov11-plate)
    set /p "CHOICE=Opcao: "
    if "!CHOICE!"=="1" set "FINAL_DIR=yolov11"
    if "!CHOICE!"=="2" set "FINAL_DIR=yolov11-plate"
)
if "!SELECTED_BACKEND!"=="SAM2" set "FINAL_DIR=segment_anything_2_image"

start "Backend !FINAL_DIR!" cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!FINAL_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && label-studio-ml start ."
pause
goto menu

:auto_label
if "!SELECTED_BACKEND!"=="" (
    echo Qual modelo deseja iniciar no backend?
    echo [1] SAM2 
    echo [2] YOLO 
    set /p "CHOICE=Opcao: "
    if "!CHOICE!"=="1" set "SELECTED_BACKEND=SAM2" 
    if "!CHOICE!"=="2" set "SELECTED_BACKEND=YOLO"
    echo SELECTED_BACKEND=!SELECTED_BACKEND!>> "%ENV_FILE%"
)

if "%LEGACY_TOKEN%"=="" (
    echo.
    set /p "TOKEN=[CONFIGURACAO] Informe o LEGACY TOKEN do Label Studio: "
    if not "!TOKEN!"=="" (
        set "LEGACY_TOKEN=!TOKEN!"
        echo LEGACY_TOKEN=!LEGACY_TOKEN!>> "%ENV_FILE%"
    ) else ( goto auto_label )
)

if "%PERSONAL_TOKEN%"=="" (
    echo.
    set /p "TOKEN=[CONFIGURACAO] Informe o PERSONAL TOKEN do Label Studio: "
    if not "!TOKEN!"=="" (
        set "PERSONAL_TOKEN=!TOKEN!"
        echo PERSONAL_TOKEN=!PERSONAL_TOKEN!>> "%ENV_FILE%"
    ) else ( goto auto_label )
)

set "RUN_BACKEND_DIR=!SELECTED_BACKEND!"
if /i "!RUN_BACKEND_DIR!"=="YOLO" set "RUN_BACKEND_DIR=yolov11-plate"

if /i "!SELECTED_BACKEND!"=="SAM2" goto auto_label_sam2
if /i "!SELECTED_BACKEND!"=="YOLO" goto auto_label_yolo

:auto_label_yolo
call :verify_yolo_models
echo Iniciando Auto-labeling CLI (YOLOv11)...
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\!RUN_BACKEND_DIR!"
%PYTHON% auto_label_cli.py --model_path="%CD%\..\..\..\..\!YOLO_PLATE_MODEL_PATH!" --vehicle_model_path="%CD%\..\..\..\..\!YOLO_VEHICLE_MODEL_PATH!"
popd
goto auto_label_end

:auto_label_sam2
echo Iniciando Auto-labeling CLI (SAM2)...
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image"
%PYTHON% auto_label_cli.py
popd

:auto_label_end
pause
goto menu

:stop_servers
taskkill /FI "WINDOWTITLE eq Label Studio*" /T /F
taskkill /FI "WINDOWTITLE eq Backend SAM2*" /T /F
pause
goto menu

:verify_yolo_models
echo Verificando modelos YOLO...
:: Verificação do detector de placas 
if exist "!YOLO_PLATE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Plate detector encontrado.
) else (
    echo [AVISO] YOLO Plate detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
    %PYTHON% download_model.py
    popd
)

:: Verificação do detector de veículos [cite: 13]
if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Vehicle detector encontrado.
) else (
    echo [AVISO] YOLO Vehicle detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11"
    %PYTHON% download_model.py
    popd
)
goto :eof

:reset_env
echo Apagando configuracoes e limpando memoria...
if exist "%ENV_FILE%" del "%ENV_FILE%"
set "DEVICE=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "LABEL_STUDIO_URL=" & set "LABEL_STUDIO_ML_BACKEND_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR="
goto init_config
