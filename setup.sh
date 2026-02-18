#!/bin/bash

# ------------------------------------------------
# Verificar Python 3.11
# ------------------------------------------------
echo "Verificando Python 3.11..."

NEED_RESTART=0

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
elif command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$PY_VER" = "3.11" ]; then
        PYTHON_BIN=python3
    else
        echo "Python 3.11 nao encontrado. Instalando..."
        NEED_RESTART=1

        if command -v apt >/dev/null 2>&1; then
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y python3.11
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -S --noconfirm python
        else
            echo "Gerenciador de pacotes nao suportado."
            echo "Instale manualmente o Python 3.11."
            exit 1
        fi

        PYTHON_BIN=python3.11
    fi
else
    echo "Python nao encontrado."
    exit 1
fi

# Se instalou agora, pedir reinicio do terminal
if [ "$NEED_RESTART" = "1" ]; then
    echo
    echo "========================================"
    echo "Python 3.11 foi instalado com sucesso."
    echo
    echo "Reinicie o terminal e"
    echo "execute o setup.sh outra vez"
    echo "========================================"
    exit 0
fi

echo "Usando $PYTHON_BIN"

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