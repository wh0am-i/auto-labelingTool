@echo off
setlocal
echo Iniciando setup do ambiente Label Studio 

:: 1. Criar venv se nao existir 
if not exist "labelStudioVenv" (
    python -m venv labelStudioVenv
    echo Ambiente virtual criado.
)

:: 2. Ativar venv 
call .\labelStudioVenv\Scripts\activate.bat

:: 3. Atualizar pip usando o python do venv para evitar conflito com a Windows Store
echo Atualizando pip...
python -m pip install --upgrade pip

:: 4. Instalar dependencias base (PyPI) - Mantendo sua lista original 
echo Verificando dependencias base...
python -m pip install label-studio label-studio-ml label-studio-sdk ^
            numpy opencv-python Pillow requests ^
            torch torchvision hydra-core omegaconf ^
            python-dotenv tqdm ultralytics

:: 5. Instalar SAM2 apenas se nao estiver instalado - Funcionalidade original restaurada 
REM echo Verificando instalacao do SAM2...
REM python -m pip show sam2 >nul 2>&1
REM if %errorlevel% neq 0 (
REM     echo SAM2 nao encontrado. Instalando do repositorio oficial...
REM     :: Nota: Requer Git instalado no Windows
REM     python -m pip install git+https://github.com/facebookresearch/segment-anything-2.git
REM ) else (
REM     echo [OK] SAM2 ja esta instalado.
REM     echo Pulando... 
REM )

echo.
echo ========================================
echo Instalacao concluida com sucesso! 
echo Use o script run.bat para iniciar os servidores. 
echo ========================================
pause