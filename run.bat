@echo off
setlocal enabledelayedexpansion

:: --- LIMPEZA DE MEMÓRIA ---
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR=" & set "DEBUG_DUMP="

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
echo BACKEND PADRAO: !SELECTED_BACKEND!
echo ========================================
echo 1. Iniciar servidores
echo 2. Auto-labeling interativo
echo 3. Parar servidores
echo 4. Resetar configuracoes (.env)
echo 5. Sair
echo ========================================
set "choice="
set /p "choice=Escolha uma opcao (1-5): "

if "!choice!"=="1" goto start_servers
if "!choice!"=="2" goto auto_label
if "!choice!"=="3" goto stop_servers
if "!choice!"=="4" goto reset_env
if "!choice!"=="5" exit
goto menu

:start_servers
echo.
echo [1] Iniciar Tudo (Frontend + Backend)
echo [2] Apenas Frontend (Label Studio)
echo [3] Apenas Backend (!SELECTED_BACKEND!)
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
    start "LS-Backend" /min cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!B_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && set LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN! && label-studio-ml start ."
    echo Servidores disparados. Verifique a barra de tarefas.
)

if "%OP_START%"=="2" (
    start "LS-Frontend" /min cmd /c "call .\labelStudioVenv\Scripts\activate.bat && label-studio start"
    echo Frontend iniciado.
)

if "%OP_START%"=="3" (
    start "LS-Backend" /min cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!B_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && set LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN! && label-studio-ml start ."
    echo Backend iniciado.
)

echo.
echo Servidor configurado para: !LABEL_STUDIO_URL!
pause
goto menu

:select_backend_folder
set "FINAL_DIR=!SELECTED_BACKEND!"
if "!SELECTED_BACKEND!"=="YOLO" (
    echo.
    echo Qual modelo YOLO deseja iniciar no backend?
    echo [1] Veiculos (yolov11)
    echo [2] Placas (yolov11-plate)
    set /p "Y_CHOICE=Escolha (1-2): "
    if "!Y_CHOICE!"=="1" set "FINAL_DIR=yolov11"
    if "!Y_CHOICE!"=="2" set "FINAL_DIR=yolov11-plate"
)
if "!SELECTED_BACKEND!"=="SAM2" set "FINAL_DIR=segment_anything_2_image"

start "Backend !FINAL_DIR!" cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\!FINAL_DIR! && set DEVICE=!DEVICE! && set LABEL_STUDIO_URL=!LABEL_STUDIO_URL! && label-studio-ml start ."
pause
goto menu

:auto_label
if "!SELECTED_BACKEND!"=="" goto auto_label_backend_missing

set "RUN_BACKEND_DIR=!SELECTED_BACKEND!"
if /i "!RUN_BACKEND_DIR!"=="YOLO" set "RUN_BACKEND_DIR=yolov11-plate"

if /i "!SELECTED_BACKEND!"=="SAM2" goto auto_label_sam2
if /i "!SELECTED_BACKEND!"=="segment_anything_2_image" goto auto_label_sam2
if /i "!SELECTED_BACKEND!"=="YOLO" goto auto_label_yolo
if /i "!SELECTED_BACKEND!"=="yolov11" goto auto_label_yolo
if /i "!SELECTED_BACKEND!"=="yolov11-plate" goto auto_label_yolo
goto auto_label_sam2

:auto_label_backend_missing
echo [ERRO] Configure o backend primeiro (Opcao 4).
pause
goto menu

:auto_label_yolo
call :verify_yolo_models
echo Iniciando Auto-labeling CLI (YOLOv11)...
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\!RUN_BACKEND_DIR!"
python auto_label_cli.py --model_path="%CD%\..\..\..\..\!YOLO_PLATE_MODEL_PATH!" --vehicle_model_path="%CD%\..\..\..\..\!YOLO_VEHICLE_MODEL_PATH!"
popd
goto auto_label_end

:auto_label_sam2
echo Iniciando Auto-labeling CLI (SAM2)...
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image"
python auto_label_cli.py
popd

:auto_label_end
pause
goto menu

:: Configuração de Debug 
if "!DEBUG_DUMP!"=="" (
    set /p "IN_DBG=[CONFIGURACAO] Habilitar debug dumps (json/png)? (S/n): "
    if /I "!IN_DBG!"=="n" ( set "DEBUG_DUMP=0" ) else ( set "DEBUG_DUMP=1" )
    echo DEBUG_DUMP=!DEBUG_DUMP!>> "%ENV_FILE%"
)

if "!SELECTED_BACKEND!"=="YOLO" goto flow_yolo
goto flow_sam2

:flow_yolo
:: Configuração do diretório de modelos 
if "!MODEL_DIR!"=="" (
    set "DEFAULT_MD=%CD%\model_runs"
    set /p "IN_MD=[CONFIGURACAO] Diretorio para logs/modelos (Enter para padrao): "
    if "!IN_MD!"=="" ( set "MODEL_DIR=!DEFAULT_MD!" ) else ( set "MODEL_DIR=!IN_MD!" )
    echo MODEL_DIR=!MODEL_DIR!>> "%ENV_FILE%"
)
if not exist "!MODEL_DIR!" mkdir "!MODEL_DIR!" 

:: Caminhos dos modelos 
if "!YOLO_PLATE_MODEL_PATH!"=="" (
    set "YOLO_PLATE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\models\best.pt"
)
if "!YOLO_VEHICLE_MODEL_PATH!"=="" (
    set "YOLO_VEHICLE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11\models\yolo11x.pt" 
)

:: --- CORREÇÃO DO DOWNLOAD (pushd para evitar erro de caminho) ---
if exist "!YOLO_PLATE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Plate detector encontrado. 
) else (
    echo [AVISO] YOLO Plate detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
    python download_model.py
    popd
)

if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Vehicle detector encontrado. 
) else (
    echo [AVISO] YOLO Vehicle detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11"
    python download_model.py
    popd
)

:start_yolo_logic
echo Iniciando Auto-labeling CLI (YOLOv11)...
call ".\labelStudioVenv\Scripts\activate.bat"

:: Exportação de variáveis para o ambiente do Python 
set "DEVICE=!DEVICE!" 
set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
set "LEGACY_TOKEN=!LEGACY_TOKEN!"
set "MODEL_DIR=!MODEL_DIR!"
set "DEBUG_DUMP=!DEBUG_DUMP!"

pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
python auto_label_cli.py --model_path="%CD%\..\..\..\..\!YOLO_PLATE_MODEL_PATH!" --vehicle_model_path="%CD%\..\..\..\..\!YOLO_VEHICLE_MODEL_PATH!"
popd 
pause
goto menu

:flow_sam2
:: Fluxo para SAM2 
echo Iniciando Auto-labeling CLI (SAM2)... 
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
set "DEVICE=!DEVICE!"
set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!"

pushd ".\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image"
python auto_label_cli.py 
popd
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
    python download_model.py
    popd
)

:: Verificação do detector de veículos [cite: 13]
if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Vehicle detector encontrado.
) else (
    echo [AVISO] YOLO Vehicle detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11"
    python download_model.py
    popd
)
goto :eof

:reset_env
echo Apagando configuracoes e limpando memoria...
if exist "%ENV_FILE%" del "%ENV_FILE%"
set "DEVICE=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "LABEL_STUDIO_URL=" & set "LABEL_STUDIO_ML_BACKEND_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR=" & set "DEBUG_DUMP="
goto init_config
