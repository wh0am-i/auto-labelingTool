@echo off
:: O 'setlocal' limita as variáveis a esta execução, mas para garantir 
:: o reset total após deletar o .env, limpamos manualmente abaixo.
setlocal enabledelayedexpansion

:: --- LIMPEZA DE MEMÓRIA (Garante que não 'lembre' de execuções anteriores no mesmo terminal) ---
set "DEVICE="
set "LABEL_STUDIO_URL="
set "PERSONAL_TOKEN="
set "LEGACY_TOKEN="
set "MODEL_CHECKPOINT="
set "MODEL_CONFIG="

set "ENV_FILE=.env"

:init_config
if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"

:: Carrega variáveis do .env (se ele existir e tiver conteúdo)
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
echo        Label Studio + SAM2 Menu
echo ========================================
echo CONFIG ATUAL: %DEVICE% ^| %LABEL_STUDIO_URL%
echo ========================================
echo 1. Iniciar servidores
echo 2. Auto-labeling interativo
echo 3. Parar servidores
echo 4. Resetar configuracoes (.env)
echo 5. Sair
echo ========================================
set /p "choice=Escolha uma opcao (1-5): "

if "%choice%"=="1" goto start_servers
if "%choice%"=="2" goto auto_label
if "%choice%"=="3" goto stop_servers
if "%choice%"=="4" goto reset_env
if "%choice%"=="5" exit
goto menu

:start_servers
start "Backend SAM2" cmd /k "call .\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\ && set DEVICE=%DEVICE%&& set LABEL_STUDIO_URL=%LABEL_STUDIO_URL%&& label-studio-ml start ./segment_anything_2_image"
start "Label Studio" cmd /k "call .\labelStudioVenv\Scripts\activate.bat && label-studio start"
goto menu

:auto_label
:: Agora, se o .env foi deletado e as variáveis limpas no topo, 
:: ele OBRIGATORIAMENTE entrará nestes IFs abaixo:

if "%PERSONAL_TOKEN%"=="" (
    echo.
    set /p "IN_PT=[CONFIGURACAO] Informe o PERSONAL_TOKEN: "
    if not "!IN_PT!"=="" (
        set "PERSONAL_TOKEN=!IN_PT!"
        echo PERSONAL_TOKEN=!PERSONAL_TOKEN!>> "%ENV_FILE%"
    )
)

if "%LEGACY_TOKEN%"=="" (
    echo.
    set /p "IN_LT=[CONFIGURACAO] Informe o LEGACY_TOKEN: "
    if not "!IN_LT!"=="" (
        set "LEGACY_TOKEN=!IN_LT!"
        echo LEGACY_TOKEN=!LEGACY_TOKEN!>> "%ENV_FILE%"
    )
)

if "%MODEL_CHECKPOINT%"=="" (
    echo.
    set /p "IN_CP=[CONFIGURACAO] Caminho Checkpoint (Enter para padrao): "
    if "!IN_CP!"=="" ( set "MODEL_CHECKPOINT=label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image\checkpoints\sam2.1_hiera_large.pt" ) else ( set "MODEL_CHECKPOINT=!IN_CP!" )
    echo MODEL_CHECKPOINT=!MODEL_CHECKPOINT!>> "%ENV_FILE%"
)

if "%MODEL_CONFIG%"=="" (
    echo.
    set /p "IN_CFG=[CONFIGURACAO] Nome do Model Config (Enter para padrao): "
    if "!IN_CFG!"=="" ( set "MODEL_CONFIG=sam2.1/sam2.1_hiera_l" ) else ( set "MODEL_CONFIG=!IN_CFG!" )
    echo MODEL_CONFIG=!MODEL_CONFIG!>> "%ENV_FILE%"
)

:: Exporta e executa
set "LABEL_STUDIO_API_KEY=%PERSONAL_TOKEN%"
set "PERSONAL_TOKEN=%PERSONAL_TOKEN%"
set "LEGACY_TOKEN=%LEGACY_TOKEN%"

echo.
echo Iniciando Auto-labeling CLI...
call .\labelStudioVenv\Scripts\activate.bat
pushd .\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image
set "LABEL_STUDIO_URL=%LABEL_STUDIO_URL%"
set "DEVICE=%DEVICE%"
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
:: Limpa as variáveis da sessão atual para forçar novas perguntas
set "DEVICE=" & set "LABEL_STUDIO_URL=" & set "PERSONAL_TOKEN=" & set "LEGACY_TOKEN=" & set "MODEL_CHECKPOINT=" & set "MODEL_CONFIG="
goto init_config