#!/bin/bash

echo "Iniciando setup do ambiente Label Studio + SAM2..."

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
pip install label-studio label-studio-ml label-studio-sdk \
            numpy opencv-python Pillow requests \
            torch torchvision hydra-core omegaconf \
            python-dotenv tqdm

# Instalar SAM2 (Direto do GitHub para garantir o pacote 'sam2')
echo "Instalando SAM2 do repositório oficial..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo -e "\n========================================"
echo "Instalação concluída com sucesso!"
echo "Use 'source ./labelStudioVenv/bin/activate' para usar."
echo "========================================"