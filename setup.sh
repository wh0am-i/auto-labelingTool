#!/bin/bash

echo "Iniciando setup do ambiente Label Studio + SAM2..."

# Instalar dependências do sistema
sudo apt update && sudo apt install -y tmux

# Criar venv se não existir
if [ ! -d "labelStudioVenv" ]; then
    python3 -m venv labelStudioVenv
    echo "Ambiente virtual criado."
fi

# Ativar venv
source ./labelStudioVenv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências base (PyPI)
# O pip por padrão já pula o que já está instalado, a menos que você use --upgrade
echo "Verificando dependências base..."
pip install label-studio label-studio-ml label-studio-sdk \
            numpy opencv-python Pillow requests \
            torch torchvision hydra-core omegaconf \
            python-dotenv tqdm ultralytics

# Instalar SAM2 apenas se não estiver instalado
echo "Verificando instalação do SAM2..."
if ! pip show "SAM-2" > /dev/null 2>&1 && ! pip show "sam2" > /dev/null 2>&1; then
    echo "SAM2 não encontrado. Instalando do repositório oficial..."
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
else
    echo "✓ SAM2 já está instalado. Pulando..."
fi

echo -e "\n========================================"
echo "Instalação concluída com sucesso!"
echo "Use 'source ./labelStudioVenv/bin/activate' para usar."
echo "========================================"
