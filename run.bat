@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON=%~dp0labelStudioVenv\Scripts\python.exe"
set "ENV_FILE=2.env"

:: --- LIMPEZA DE MEMORIA DA SESSAO ---
set "LABEL_STUDIO_URL="
set "PERSONAL_TOKEN="
set "LEGACY_TOKEN="
set "YOLO_ENABLE_OPENVINO="
set "YOLO_PLATE_MODEL_PATH="
set "YOLO_VEHICLE_MODEL_PATH="

:init_config
if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"

:: Carrega variaveis do arquivo .env/.2.env
for /f "usebackq tokens=*" %%i in ("%ENV_FILE%") do (
    set "line=%%i"
    if "!line:~0,1!" neq "#" if not "!line!"=="" (
        for /f "tokens=1* delims==" %%a in ("!line!") do (
            set "k=%%a"
            set "v=%%b"
            for /f "tokens=* delims= " %%K in ("!k!") do set "k=%%K"
            for /f "tokens=* delims= " %%V in ("!v!") do set "v=%%V"
            for /l %%# in (1,1,32) do if defined v if "!v:~-1!"==" " set "v=!v:~0,-1!"
            set "!k!=!v!"
        )
    )
)

:: --- CONFIGURACAO DE PATHS ---
if "!YOLO_PLATE_MODEL_PATH!"=="" (
    echo.
    set "DEFAULT_PLATE=label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\models\best.pt"
    set /p "IN_PP=[CONFIGURACAO] Caminho do modelo yolo11-plate (Enter para padrao): "
    if "!IN_PP!"=="" ( set "YOLO_PLATE_MODEL_PATH=!DEFAULT_PLATE!" ) else ( set "YOLO_PLATE_MODEL_PATH=!IN_PP!" )
    echo YOLO_PLATE_MODEL_PATH=!YOLO_PLATE_MODEL_PATH!>> "%ENV_FILE%"
)

if "!YOLO_VEHICLE_MODEL_PATH!"=="" (
    echo.
    set "DEFAULT_VEHICLE=label-studio-ml-backend\label_studio_ml\examples\yolov11\models\yolo11x.pt"
    set /p "IN_VP=[CONFIGURACAO] Caminho do modelo yolo11x (Enter para padrao): "
    if "!IN_VP!"=="" ( set "YOLO_VEHICLE_MODEL_PATH=!DEFAULT_VEHICLE!" ) else ( set "YOLO_VEHICLE_MODEL_PATH=!IN_VP!" )
    echo YOLO_VEHICLE_MODEL_PATH=!YOLO_VEHICLE_MODEL_PATH!>> "%ENV_FILE%"
)

:: --- CONFIGURACAO DE URL ---
if "%LABEL_STUDIO_URL%"=="" (
    echo.
    set /p "IN_URL=[CONFIGURACAO] URL do Label Studio (Enter para padrao: http://localhost:8080): "
    if "!IN_URL!"=="" ( set "LABEL_STUDIO_URL=http://localhost:8080" ) else ( set "LABEL_STUDIO_URL=!IN_URL!" )
    echo LABEL_STUDIO_URL=!LABEL_STUDIO_URL!>> "%ENV_FILE%"
)

:menu
cls
echo ==========================================================
echo         Label Studio Auto-Labeling Tool (Windows)
echo ==========================================================
echo CONFIG FRONTEND: %LABEL_STUDIO_URL%
echo ========================================
echo 1. Iniciar servidor
echo 2. Auto-labeling interativo (YOLO)
echo 3. Parar servidor
echo 4. Resetar configuracoes (2.env)
echo 5. Sair
echo ========================================
set "CHOICE="
set /p "CHOICE=Escolha uma opcao: "

if "!CHOICE!"=="1" goto start_server
if "!CHOICE!"=="2" goto auto_label
if "!CHOICE!"=="3" goto stop_servers
if "!CHOICE!"=="4" goto reset_env
if "!CHOICE!"=="5" exit /b 0
goto menu

:start_server
echo.
echo Iniciando Frontend (Label Studio)...
start "LS-Frontend" /min cmd /c "call .\labelStudioVenv\Scripts\activate.bat && label-studio start"
echo Frontend iniciado em !LABEL_STUDIO_URL!
echo.
pause
goto menu

:auto_label
if "%LEGACY_TOKEN%"=="" (
    :ask_legacy
    echo.
    set /p "TOKEN=[CONFIGURACAO] Informe o LEGACY TOKEN do Label Studio: "
    if "!TOKEN!"=="" (
        echo [ERRO] O token nao pode estar vazio.
        goto ask_legacy
    )
    set "LEGACY_TOKEN=!TOKEN!"
    echo LEGACY_TOKEN=!LEGACY_TOKEN!>> "%ENV_FILE%"
)

if "%PERSONAL_TOKEN%"=="" (
    :ask_personal
    echo.
    set /p "TOKEN=[CONFIGURACAO] Informe o PERSONAL TOKEN do Label Studio: "
    if "!TOKEN!"=="" (
        echo [ERRO] O token nao pode estar vazio.
        goto ask_personal
    )
    set "PERSONAL_TOKEN=!TOKEN!"
    echo PERSONAL_TOKEN=!PERSONAL_TOKEN!>> "%ENV_FILE%"
)

call :verify_yolo_models
echo.
echo Iniciando Auto-labeling CLI (YOLOv11)...
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
%PYTHON% auto_label_cli.py --model_path="%CD%\..\..\..\..\!YOLO_PLATE_MODEL_PATH!" --vehicle_model_path="%CD%\..\..\..\..\!YOLO_VEHICLE_MODEL_PATH!"
popd
echo.
pause
goto menu

:stop_servers
echo.
echo Encerrando processos do Label Studio...
taskkill /FI "WINDOWTITLE eq LS-Frontend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Label Studio*" /T /F >nul 2>&1
echo Concluido.
echo.
pause
goto menu

:verify_yolo_models
if "!YOLO_ENABLE_OPENVINO!"=="" (
    :ask_openvino
    echo.
    set /p "IN_OV=[CONFIGURACAO] Deseja ativar execucao com openVINO? [S/n]: "
    if "!IN_OV!"=="" (
        set "YOLO_ENABLE_OPENVINO=1"
    ) else if /I "!IN_OV!"=="S" (
        set "YOLO_ENABLE_OPENVINO=1"
    ) else if /I "!IN_OV!"=="N" (
        set "YOLO_ENABLE_OPENVINO=0"
    ) else (
        echo [AVISO] Opcao invalida.
        goto ask_openvino
    )
    echo YOLO_ENABLE_OPENVINO=!YOLO_ENABLE_OPENVINO!>> "%ENV_FILE%"
)

echo.
echo [DEBUG] Verificando arquivos dos modelos...

if exist "!YOLO_PLATE_MODEL_PATH!" (
    echo [DEBUG] Modelo yolo11-plate encontrado.
) else (
    echo [AVISO] Modelo yolo11-plate NAO encontrado.
    :ask_download_plate
    set /p "DL_P=Deseja baixar o modelo yolo11-plate agora? [S/n]: "
    if "!DL_P!"=="" set "DL_P=S"
    if /I "!DL_P!"=="S" (
        pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
        %PYTHON% download_model.py
        popd
    ) else if /I "!DL_P!"=="N" (
        echo [AVISO] Continuando sem baixar. Erros podem ocorrer na predicao.
    ) else (
        goto ask_download_plate
    )
)

if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [DEBUG] Modelo yolo11x encontrado.
) else (
    echo [AVISO] Modelo yolo11x NAO encontrado.
    :ask_download_veh
    set /p "DL_V=Deseja baixar o modelo yolo11x agora? [S/n]: "
    if "!DL_V!"=="" set "DL_V=S"
    if /I "!DL_V!"=="S" (
        pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11"
        %PYTHON% download_model.py
        popd
    ) else if /I "!DL_V!"=="N" (
        echo [AVISO] Continuando sem baixar.
    ) else (
        goto ask_download_veh
    )
)
goto :eof

:reset_env
echo Apagando configuracoes e limpando memoria...
if exist "%ENV_FILE%" del "%ENV_FILE%"
set "YOLO_PLATE_MODEL_PATH="
set "YOLO_VEHICLE_MODEL_PATH="
set "LABEL_STUDIO_URL="
set "PERSONAL_TOKEN="
set "LEGACY_TOKEN="
set "YOLO_ENABLE_OPENVINO="
goto init_config
