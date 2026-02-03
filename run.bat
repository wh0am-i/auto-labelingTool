@echo off
setlocal enabledelayedexpansion

:: Nome da sessão/janela
set SESSION_NAME=LabelStudio_SAM

:menu
cls
echo ========================================
echo        Label Studio + SAM2 Menu (Windows)
echo ========================================
echo 1. Iniciar servidores
echo 2. Auto-labeling interativo
echo 3. Parar servidores (Fecha as janelas)
echo 4. Sair
echo ========================================
set /p choice="Escolha uma opcao (1-4): "

if "%choice%"=="1" goto iniciar
if "%choice%"=="2" goto autolabel
if "%choice%"=="3" goto parar
if "%choice%"=="4" goto sair
goto menu

:iniciar
:: VERIFICAÇÃO DA VENV
if not exist ".\labelStudioVenv\" (
    echo.
    echo ERRO: A pasta 'labelStudioVenv' nao existe no diretorio atual.
    echo Crie o ambiente virtual com: python -m venv labelStudioVenv
    pause
    goto menu
)

echo Criando janelas dos servidores...

:: Setup Backend
start "LS-Backend-SAM" cmd /k ".\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\ && set DEVICE=cpu && set LABEL_STUDIO_URL=http://127.0.0.1:8080 && label-studio-ml start ./segment_anything_2_image"

:: Setup Frontend
start "LS-Frontend" cmd /k ".\labelStudioVenv\Scripts\activate.bat && label-studio start"

echo Servidores iniciados em novas janelas.
pause
goto menu

:autolabel
:: VERIFICAÇÃO DA VENV
if not exist ".\labelStudioVenv\" (
    echo ERRO: A pasta 'labelStudioVenv' nao existe.
    pause
    goto menu
)

if "%LABEL_STUDIO_API_KEY%"=="" (
    set /p API_KEY="Insira seu Token do Label Studio: "
) else (
    set API_KEY=%LABEL_STUDIO_API_KEY%
)

echo Iniciando Auto-labeling...
.\labelStudioVenv\Scripts\activate.bat && cd .\label-studio-ml-backend\label_studio_ml\examples\segment_anything_2_image && set LABEL_STUDIO_URL=http://127.0.0.1:8080 && set DEVICE=cpu && python auto_label_cli.py
pause
goto menu

:parar
echo Tentando fechar processos do Label Studio...
taskkill /FI "WINDOWTITLE eq LS-Backend-SAM*" /F
taskkill /FI "WINDOWTITLE eq LS-Frontend*" /F
echo Servidores finalizados.
pause
goto menu

:sair
exit