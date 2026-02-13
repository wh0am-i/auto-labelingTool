#!/bin/bash

echo "Iniciando setup do ambiente Label Studio" 

# Instalar dependências do sistema
sudo apt update && sudo apt install -y tmux 

# Criar venv se não existir
if [ ! -d "labelStudioVenv" ]; then 
    python3 -m venv labelStudioVenv 
    echo "Ambiente virtual criado." 
fi

# Ativar venv
source ./labelStudioVenv/bin/activate 

# Atualizar pip usando o executável do venv
echo "Atualizando pip..."
python3 -m pip install --upgrade pip 

# Instalar dependências base (PyPI)
# Usamos 'python3 -m pip' para garantir que o binário do venv seja o destino
echo "Verificando dependências base..." 
python3 -m pip install label-studio label-studio-ml label-studio-sdk \
            numpy opencv-python Pillow requests \
            torch torchvision hydra-core omegaconf \
            python-dotenv tqdm ultralytics 

# Instalar SAM2 apenas se não estiver instalado
# echo "Verificando instalação do SAM2..." 
# if ! python3 -m pip show "sam2" > /dev/null 2>&1; then 
#     echo "SAM2 não encontrado. Instalando do repositório oficial..." 
    # Certifique-se de ter o git instalado: sudo apt install git
#    python3 -m pip install git+https://github.com/facebookresearch/segment-anything-2.git 
# else
#     echo "✓ SAM2 já está instalado. Pulando..." 
# fi

echo -e "\n========================================" 
echo "Instalação concluída com sucesso!" 
echo "Use o script run.sh para iniciar os servidores" 
echo "========================================" 