@echo off
setlocal enabledelayedexpansion

:: --- LIMPEZA DE MEMÓRIA ---
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND="

set "ENV_FILE=.env"

:init_config
if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"

for /f "usebackq tokens=*" %%i in ("%ENV_FILE%") do (
    set "line=%%i"
    if "!line:~0,1!" neq "#" if not "!line!"=="" (
        for /f "tokens=1,2 delims==" %%a in ("!line!") do set "%%a=%%b"
    )
)

:: --- CONFIGURAÇÃO INICIAL ---
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
    set /p "IN_URL=[CONFIGURACAO] URL do Label Studio (Padrao: http://127.0.0.1:8080): "
    if "!IN_URL!"=="" ( set "LABEL_STUDIO_URL=http://127.0.0.1:8080" ) else ( set "LABEL_STUDIO_URL=!IN_URL!" )
    echo LABEL_STUDIO_URL=!LABEL_STUDIO_URL!>> "%ENV_FILE%"
)

:menu
cls
echo ========================================
echo         Label Studio + SAM2 Menu
echo ========================================
echo CONFIG ATUAL: %DEVICE% ^| %LABEL_STUDIO_URL%
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
start "Backend SAM2" cmd /k "call ".\labelStudioVenv\Scripts\activate.bat" && cd ".\label-studio-ml-backend\label_studio_ml\examples\" && set "DEVICE=!DEVICE!" && set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!" && label-studio-ml start ./segment_anything_2_image"
start "Label Studio" cmd /k "call ".\labelStudioVenv\Scripts\activate.bat" && label-studio start"
goto menu

:auto_label
:: Tokens
if "!PERSONAL_TOKEN!"=="" (
    set /p "IN_PT=[CONFIGURACAO] Informe o PERSONAL_TOKEN do Label Studio: "
    if not "!IN_PT!"=="" ( set "PERSONAL_TOKEN=!IN_PT!" & echo PERSONAL_TOKEN=!PERSONAL_TOKEN!>> "%ENV_FILE%" )
)
if "!LEGACY_TOKEN!"=="" (
    set /p "IN_LT=[CONFIGURACAO] Informe o LEGACY_TOKEN do Label Studio: "
    if not "!IN_LT!"=="" ( set "LEGACY_TOKEN=!IN_LT!" & echo LEGACY_TOKEN=!LEGACY_TOKEN!>> "%ENV_FILE%" )
)

:: Seleção de Backend
if "!SELECTED_BACKEND!"=="" (
    echo.
    set "IN_BACK=1"
    set /p "IN_BACK=[CONFIGURACAO] Escolha backend: 1) SAM2  2) YOLOv11 (Enter = 1): "
    if "!IN_BACK!"=="2" ( set "SELECTED_BACKEND=YOLO" ) else ( set "SELECTED_BACKEND=SAM2" )
    echo SELECTED_BACKEND=!SELECTED_BACKEND!>> "%ENV_FILE%"
)

:: Pergunta/define se debug deve estar ativo (grava em .env)
if "!DEBUG_DUMP!"=="" (
    set /p "IN_DBG=[CONFIGURACAO] Habilitar debug dumps (json/png)? (S/n): "
    if /I "!IN_DBG!"=="n" (
        set "DEBUG_DUMP=0"
    ) else (
        set "DEBUG_DUMP=1"
    )
    echo DEBUG_DUMP=!DEBUG_DUMP!>> "%ENV_FILE%"
)
set "DEBUG_DUMP=!DEBUG_DUMP!"

if "!SELECTED_BACKEND!"=="YOLO" goto flow_yolo
goto flow_sam2

:flow_yolo
:: --- CONFIGURAÇÃO YOLO PLATE ---
if "!YOLO_PLATE_MODEL_PATH!"=="" (
    set /p "IN_YMP=[CONFIGURACAO] Caminho YOLOv11 Plate Model (Enter para padrao): "
    if "!IN_YMP!"=="" (
        set "YOLO_PLATE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\models\best.pt"
    ) else (
        set "YOLO_PLATE_MODEL_PATH=!IN_YMP!"
    )
    echo YOLO_PLATE_MODEL_PATH=!YOLO_PLATE_MODEL_PATH!>> "%ENV_FILE%"
)

:: --- CONFIGURAÇÃO YOLO VEHICLE (YOLOv11x) ---
if "!YOLO_VEHICLE_MODEL_PATH!"=="" (
    set /p "IN_YVP=[CONFIGURACAO] Caminho YOLOv11 Vehicle Model (Enter para padrao): "
    if "!IN_YVP!"=="" (
        set "YOLO_VEHICLE_MODEL_PATH=label-studio-ml-backend\label_studio_ml\examples\yolov11\models\yolo11x.pt"
    ) else (
        set "YOLO_VEHICLE_MODEL_PATH=!IN_YVP!"
    )
    echo YOLO_VEHICLE_MODEL_PATH=!YOLO_VEHICLE_MODEL_PATH!>> "%ENV_FILE%"
)

:: Verificação de arquivos (sem tentativa de download automatica)
if exist "!YOLO_PLATE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Plate detector encontrado em: !YOLO_PLATE_MODEL_PATH!
) else (
    echo [AVISO] YOLO Plate detector nao encontrado em: !YOLO_PLATE_MODEL_PATH!
    echo [AVISO] Tentando download...
    python3 .\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate\download_model.py
)

if exist "!YOLO_VEHICLE_MODEL_PATH!" (
    echo [SUCESSO] YOLO Vehicle detector encontrado em: !YOLO_VEHICLE_MODEL_PATH!
) else (
    echo [AVISO] YOLO Vehicle detector nao encontrado em: !YOLO_VEHICLE_MODEL_PATH!
    echo [AVISO] Tentando download...
    python3 .\label-studio-ml-backend\label_studio_ml\examples\yolov11\download_model.py
)

:start_yolo_logic
echo Iniciando Auto-labeling CLI (YOLOv11)...
call ".\labelStudioVenv\Scripts\activate.bat"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\yolov11-plate"
set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!"
set "DEVICE=!DEVICE!"
python auto_label_cli.py --model_path="!YOLO_PLATE_MODEL_PATH!" --vehicle_model_path="!YOLO_VEHICLE_MODEL_PATH!"
popd
pause
goto menu

:flow_sam2
if "!MODEL_CHECKPOINT!"=="" (
    set /p "IN_CP=[CONFIGURACAO] Caminho Checkpoint (Enter para padrao): "
    if "!IN_CP!"=="" ( set "MODEL_CHECKPOINT=label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image\checkpoints\sam2.1_hiera_large.pt" ) else ( set "MODEL_CHECKPOINT=!IN_CP!" )
    echo MODEL_CHECKPOINT=!MODEL_CHECKPOINT!>> "%ENV_FILE%"
)
if "!MODEL_CONFIG!"=="" (
    set /p "IN_CFG=[CONFIGURACAO] Nome do Model Config (Enter para padrao): "
    if "!IN_CFG!"=="" ( set "MODEL_CONFIG=sam2.1/sam2.1_hiera_l" ) else ( set "MODEL_CONFIG=!IN_CFG!" )
    echo MODEL_CONFIG=!MODEL_CONFIG!>> "%ENV_FILE%"
)
set "LABEL_STUDIO_API_KEY=!PERSONAL_TOKEN!"
echo Iniciando Auto-labeling CLI (SAM2)...
call ".\labelStudioVenv\Scripts\activate.bat"
pushd ".\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image"
set "LABEL_STUDIO_URL=!LABEL_STUDIO_URL!"
set "DEVICE=!DEVICE!"
python auto_label_cli.py
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
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG=" & set "YOLO_PLATE_MODEL_PATH=" & set "YOLO_VEHICLE_MODEL_PATH=" & set "SELECTED_BACKEND="
goto init_config