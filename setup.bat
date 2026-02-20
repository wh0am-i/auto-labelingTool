@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ========================================
echo  Setup isolado do Label Studio
echo ========================================

:: Pasta base do projeto (sempre relativa ao .bat)
set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"

:: ------------------------------------------------
:: Verificar Python 3.11
:: ------------------------------------------------
echo Verificando Python 3.11...

py -3.11 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python 3.11 nao encontrado.
    echo Instalando Python 3.11 via winget...

    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements

    echo.
    echo Verificando instalacao...

    py -3.11 --version >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo ========================================
        echo Python 3.11 foi instalado,
        echo mas o terminal precisa ser reiniciado
        echo para atualizar o PATH.
        echo.
        echo Reinicie o terminal e
        echo execute o setup.bat outra vez.
        echo ========================================
        pause
        exit /b 0
    ) ELSE (
        echo Instalacao concluida com sucesso.
    )

) ELSE (
    echo Python 3.11 encontrado.
)

set "VENV_DIR=%BASE_DIR%labelStudioVenv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

:: 1. Criar venv se nao existir
if not exist "%PYTHON_EXE%" (
    echo Criando ambiente virtual...
    py -3.11 -m venv "%VENV_DIR%"
    
    if errorlevel 1 (
        echo ERRO ao criar venv.
        pause
        exit /b 1
    )

    echo Ambiente virtual criado com sucesso.
)

:: 2. Garantir que estamos usando o Python da venv
if not exist "%PYTHON_EXE%" (
    echo ERRO: Python da venv nao encontrado.
    pause
    exit /b 1
)

:: 3. Atualizar pip da venv (forcado)
echo Atualizando pip da venv...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo ERRO ao atualizar pip.
    pause
    exit /b 1
)

:: 4. Instalar dependencias
echo Instalando dependencias...

"%PYTHON_EXE%" -m pip install tensorrt openvino label-studio label-studio-ml label-studio-sdk numpy opencv-python Pillow requests torch torchvision hydra-core omegaconf python-dotenv tqdm ultralytics

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERRO na instalacao das dependencias.
    echo Verifique mensagens acima.
    echo ========================================
    pause
    exit /b 1
)

:: 5. Verificacao final real
echo.
echo Verificando instalacao do Label Studio...

"%PYTHON_EXE%" -c "import label_studio" 2>nul
if errorlevel 1 (
    echo ERRO: Label Studio nao foi instalado corretamente.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Instalacao concluida com sucesso!
echo Utilize o run.bat para iniciar
echo ========================================
pause
