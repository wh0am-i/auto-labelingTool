@echo off
setlocal enabledelayedexpansion

:: --- LIMPEZA DE MEMÓRIA ---
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR=" & set "DEBUG_DUMP="

set "ENV_FILE=.env"

:init_config
if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"

:: Carrega variáveis do arquivo .env [cite: 2]
for /f "usebackq tokens=*" %%i in ("%ENV_FILE%") do (
    set "line=%%i"
    if "!line:~0,1!" neq "#" if not "!line!"=="" (
        for /f "tokens=1,2 delims==" %%a in ("!line!") do set "%%a=%%b"
    )
)

:: --- CONFIGURAÇÃO INICIAL ---
if "%DEVICE%"=="" (
    echo.
    set /p "IN_DEV=[CONFIGURACAO] Dispositivo (cpu / cuda): " [cite: 3]
    if not "!IN_DEV!"=="" (
        set "DEVICE=!IN_DEV!"
        echo DEVICE=!DEVICE!>> "%ENV_FILE%"
    ) else ( goto init_config )
)

if "%LABEL_STUDIO_URL%"=="" (
    echo.
    set /p "IN_URL=[CONFIGURACAO] URL do Label Studio (Padrao: http://127.0.0.1:8080): "
    if "!IN_URL!"=="" ( set "LABEL_STUDIO_URL=http://127.0.0.1:8080" ) else ( set "LABEL_STUDIO_URL=!IN_URL!" )
    echo LABEL_STUDIO_URL=!LABEL_STUDIO_URL!>> "%ENV_FILE%"
)

:menu
cls
echo ========================================
echo         Label Studio + SAM2 Menu
echo ========================================
echo CONFIG ATUAL: %DEVICE% ^| %LABEL_STUDIO_URL% [cite: 4]
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
:: Inicia os processos em janelas separadas [cite: 2]
start "Backend SAM2" cmd /k "call ".\labelStudioVenv\Scripts\activate.bat" && cd ".\label-studio-ml-backend\label_studio_ml\examples\" && set "DEVICE=!DEVICE!" && set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!" && label-studio-ml start ./segment_anything_2_image"
start "Label Studio" cmd /k "call ".\labelStudioVenv\Scripts\activate.bat" && label-studio start"
goto menu

:auto_label
:: --- TOKENS (Garantindo que serão salvos e exportados) ---
if "!PERSONAL_TOKEN!"=="" (
    set /p "IN_PT=[CONFIGURACAO] Informe o PERSONAL_TOKEN do Label Studio: "
    if not "!IN_PT!"=="" ( 
        set "PERSONAL_TOKEN=!IN_PT!" [cite: 5]
        echo PERSONAL_TOKEN=!PERSONAL_TOKEN!>> "%ENV_FILE%"
    )
)
if "!LEGACY_TOKEN!"=="" (
    set /p "IN_LT=[CONFIGURACAO] Informe o LEGACY_TOKEN do Label Studio: "
    if not "!IN_LT!"=="" ( 
        set "LEGACY_TOKEN=!IN_LT!"
        echo LEGACY_TOKEN=!LEGACY_TOKEN!>> "%ENV_FILE%"
    )
)

:: Seleção de Backend [cite: 6]
if "!SELECTED_BACKEND!"=="" (
    echo.
    set /p "IN_BACK=[CONFIGURACAO] Escolha backend: 1) SAM2  2) YOLOv11 (Enter = 1): "
    if "!IN_BACK!"=="2" ( set "SELECTED_BACKEND=YOLO" ) else ( set "SELECTED_BACKEND=SAM2" )
    echo SELECTED_BACKEND=!SELECTED_BACKEND!>> "%ENV_FILE%"
)

:: Configuração de Debug [cite: 7]
if "!DEBUG_DUMP!"=="" (
    set /p "IN_DBG=[CONFIGURACAO] Habilitar debug dumps (json/png)? (S/n): "
    if /I "!IN_DBG!"=="n" ( set "DEBUG_DUMP=0" ) else ( set "DEBUG_DUMP=1" )
    echo DEBUG_DUMP=!DEBUG_DUMP!>> "%ENV_FILE%"
)

if "!SELECTED_BACKEND!"=="YOLO" goto flow_yolo
goto flow_sam2

:flow_yolo
:: Configuração do diretório de modelos [cite: 8]
if "!MODEL_DIR!"=="" (
    set "DEFAULT_MD=%CD%\model_runs"
    set /p "IN_MD=[CONFIGURACAO] Diretorio para logs/modelos (Enter para padrao): "
    if "!IN_MD!"=="" ( set "MODEL_DIR=!DEFAULT_MD!" ) else ( set "MODEL_DIR=!IN_MD!" )
    echo MODEL_DIR=!MODEL_DIR!>> "%ENV_FILE%"
)
if not exist "!MODEL_DIR!" mkdir "!MODEL_DIR!" [cite: 8]

:: Caminhos dos modelos [cite: 9]
if "!YOLO_PLATE_MODEL_PATH!"=="" (
    set "YOLO_PLATE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\models\best.pt"
)
if "!YOLO_VEHICLE_MODEL_PATH!"=="" (
    set "YOLO_VEHICLE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11\models\yolo11x.pt" [cite: 9]
)

:: --- CORREÇÃO DO DOWNLOAD (pushd para evitar erro de caminho) ---
if exist "!YOLO_PLATE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Plate detector encontrado. [cite: 10]
) else (
    echo [AVISO] YOLO Plate detector nao encontrado. Baixando...
    pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
    python download_model.py
    popd
)

if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Vehicle detector encontrado. [cite: 11]
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
:: Fluxo para SAM2 [cite: 14, 15]
echo Iniciando Auto-labeling CLI (SAM2)... [cite: 14]
call ".\labelStudioVenv\Scripts\activate.bat"
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
set "DEVICE=!DEVICE!"
set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!"

pushd ".\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image"
python auto_label_cli.py [cite: 15]
popd
pause
goto menu

:stop_servers
taskkill /FI "WINDOWTITLE eq Label Studio*" /T /F
taskkill /FI "WINDOWTITLE eq Backend SAM2*" /T /F
pause
goto menu

:reset_env
echo Apagando configuracoes e limpando memoria...
if exist "%ENV_FILE%" del "%ENV_FILE%"
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND=" & set "MODEL_DIR=" & set "DEBUG_DUMP="
goto init_config