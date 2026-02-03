@echo off
echo Iniciando setup do ambiente Label Studio + SAM2...

:: Criar venv se nao existir
if not exist "labelStudioVenv" (
    python -m venv labelStudioVenv
    echo Ambiente virtual criado.
)

:: Ativar venv
call .\labelStudioVenv\Scripts\activate.bat

:: Atualizar pip
python -m pip install --upgrade pip

:: Instalar dependencias base (PyPI)
echo Instalando dependencias base...
pip install label-studio label-studio-ml label-studio-sdk ^
            numpy opencv-python Pillow requests ^
            torch torchvision hydra-core omegaconf ^
            python-dotenv tqdm

:: Instalar SAM2
echo Instalando SAM2 do repositorio oficial...
pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo.
echo ========================================
echo Instalacao concluida com sucesso!
echo Use o script de menu para iniciar os servidores.
echo ========================================
pause