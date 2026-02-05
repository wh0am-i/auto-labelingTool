@echo off
setlocal
echo Iniciando setup do ambiente Label Studio + SAM2...

:: Criar venv se nao existir
if not exist "labelStudioVenv" (
    python -m venv labelStudioVenv
    echo Ambiente virtual criado.
)

:: Ativar venv
call .\labelStudioVenv\Scripts\activate.bat

:: Atualizar pip
echo Atualizando pip...
python -m pip install --upgrade pip

:: Instalar dependencias base (PyPI)
echo Verificando dependencias base...
:: O pip ja ignora o que ja esta instalado por padrao
pip install label-studio label-studio-ml label-studio-sdk ^
            numpy opencv-python Pillow requests ^
            torch torchvision hydra-core omegaconf ^
            python-dotenv tqdm ultralytics

:: Instalar SAM2 apenas se nao estiver instalado
echo Verificando instalacao do SAM2...
pip show sam2 >nul 2>&1
if %errorlevel% neq 0 (
    echo SAM2 nao encontrado. Instalando do repositorio oficial...
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
) else (
    echo [OK] SAM2 ja esta instalado. Pulando...
)

echo.
echo ========================================
echo Instalacao concluida com sucesso!
echo Use o script de menu para iniciar os servidores.
echo ========================================
pause
