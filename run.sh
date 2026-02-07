#!/bin/bash

# --- CONFIGURAÇÃO ---
ENV_FILE=".env"

# Função para carregar variáveis do .env
load_env() {
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
}

# Função para limpar memória (variáveis da sessão)
reset_memory() {
    unset DEVICE LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN MODEL_CHECKPOINT MODEL_CONFIG
}

init_config() {
    [ ! -f "$ENV_FILE" ] && touch "$ENV_FILE"
    load_env

    # Configuração do Dispositivo
    if [ -z "$DEVICE" ]; then
        read -p "[CONFIGURACAO] Dispositivo (cpu / cuda): " IN_DEV
        if [ ! -z "$IN_DEV" ]; then
            DEVICE=$IN_DEV
            echo "DEVICE=$DEVICE" >> "$ENV_FILE"
        else
            init_config
        fi
    fi

    # Configuração da URL
    if [ -z "$LABEL_STUDIO_URL" ]; then
        read -p "[CONFIGURACAO] URL do Label Studio (Padrao: http://127.0.0.1:8080): " IN_URL
        if [ -z "$IN_URL" ]; then
            LABEL_STUDIO_URL="http://127.0.0.1:8080"
        else
            LABEL_STUDIO_URL=$IN_URL
        fi
        echo "LABEL_STUDIO_URL=$LABEL_STUDIO_URL" >> "$ENV_FILE"
    fi
}

start_servers() {
    SESSION_NAME="labelstudio"
    BASE_DIR="$(pwd)"

    echo "Abrindo nova janela de terminal com tmux (2 panes)..."

    if ! command -v gnome-terminal &>/dev/null; then
        echo "gnome-terminal não encontrado."
        return 1
    fi

    export BASE_DIR
    export DEVICE
    export LABEL_STUDIO_URL
    export PERSONAL_TOKEN
    export LEGACY_TOKEN

    gnome-terminal -- bash -c "
        cd \"$BASE_DIR\" || exit 1
        source ./labelStudioVenv/bin/activate

        # Se a sessão já existir, apenas reanexa
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            tmux attach -t $SESSION_NAME
            exec bash
        fi

        # Cria sessão e divide em 2 panes
        tmux new-session -d -s $SESSION_NAME

        # Split vertical (lado a lado)
        tmux split-window -h -t $SESSION_NAME

        # Pane 0 - Frontend (Label Studio)
        tmux send-keys -t $SESSION_NAME:0.0 \"
            label-studio start
        \" C-m

        # Pane 1 - Backend (SAM2)
        tmux send-keys -t $SESSION_NAME:0.1 \"
            export DEVICE=$DEVICE
            export LABEL_STUDIO_URL=$LABEL_STUDIO_URL
            export PERSONAL_TOKEN=$PERSONAL_TOKEN
            export LEGACY_TOKEN=$LEGACY_TOKEN
            cd ./label-studio-ml-backend/label_studio_ml/examples/
            label-studio-ml start ./segment_anything_2_image
        \" C-m

        tmux attach -t $SESSION_NAME
        exec bash
    "
}

auto_label() {
    load_env
    
    if [ -z "$PERSONAL_TOKEN" ]; then
        read -p "[CONFIGURACAO] Informe o PERSONAL_TOKEN: " IN_PT
        [ ! -z "$IN_PT" ] && PERSONAL_TOKEN=$IN_PT && echo "PERSONAL_TOKEN=$PERSONAL_TOKEN" >> "$ENV_FILE"
    fi

    if [ -z "$LEGACY_TOKEN" ]; then
        read -p "[CONFIGURACAO] Informe o LEGACY_TOKEN: " IN_LT
        [ ! -z "$IN_LT" ] && LEGACY_TOKEN=$IN_LT && echo "LEGACY_TOKEN=$LEGACY_TOKEN" >> "$ENV_FILE"
    fi

    # Escolha de backend (SAM2 ou YOLOv11)
    echo
    read -p "[CONFIGURACAO] Escolha backend: 1) SAM2  2) YOLOv11  (Enter = 1): " IN_BACK
    BACKEND=${IN_BACK:-1}

    export LABEL_STUDIO_API_KEY=$PERSONAL_TOKEN
    export PERSONAL_TOKEN=$PERSONAL_TOKEN
    export LEGACY_TOKEN=$LEGACY_TOKEN
    export DEVICE=$DEVICE
    export LABEL_STUDIO_URL=$LABEL_STUDIO_URL

    if [ "$BACKEND" = "2" ] || [ "$BACKEND" = "yolo" ] || [ "$BACKEND" = "YOLO" ]; then
        # Ensure YOLO model is present (attempt download if missing)
        export YOLO_MODEL_PATH=${YOLO_MODEL_PATH:-"label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/models/best.pt"}
        if [ ! -f "$YOLO_MODEL_PATH" ]; then
            echo "YOLO model não encontrado em $YOLO_MODEL_PATH. Tentando baixar..."
            source ./labelStudioVenv/bin/activate
            python3 ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/download_yolo_model.py
        fi

        echo "Iniciando Auto-labeling CLI (YOLOv11)..."
        source ./labelStudioVenv/bin/activate
        pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate > /dev/null
        python3 auto_label_cli.py
        popd > /dev/null
        read -p "Pressione Enter para continuar..."
    else
        # Default: SAM2 flow
        if [ -z "$MODEL_CHECKPOINT" ]; then
            read -p "[CONFIGURACAO] Caminho Checkpoint (Enter para padrao): " IN_CP
            MODEL_CHECKPOINT=${IN_CP:-"label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/checkpoints/sam2.1_hiera_large.pt"}
            echo "MODEL_CHECKPOINT=$MODEL_CHECKPOINT" >> "$ENV_FILE"
        fi

        if [ -z "$MODEL_CONFIG" ]; then
            read -p "[CONFIGURACAO] Nome do Model Config (Enter para padrao): " IN_CFG
            MODEL_CONFIG=${IN_CFG:-"sam2.1/sam2.1_hiera_l"}
            echo "MODEL_CONFIG=$MODEL_CONFIG" >> "$ENV_FILE"
        fi

        echo "Iniciando Auto-labeling CLI (SAM2)..."
        source ./labelStudioVenv/bin/activate
        pushd ./label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image > /dev/null
        python3 auto_label_cli.py
        popd > /dev/null
        read -p "Pressione Enter para continuar..."
    fi
}

stop_servers() {
    echo "Parando servidores..."
    pkill -f "label-studio"
    pkill -f "label-studio-ml"
    echo "Processos encerrados."
    sleep 2
}

reset_env() {
    echo "Apagando configuracoes e limpando memoria..."
    [ -f "$ENV_FILE" ] && rm "$ENV_FILE"
    reset_memory
    init_config
}

# Loop do Menu Principal
while true; do
    init_config
    clear
    echo "========================================"
    echo "       Label Studio + SAM2 Menu (Linux)"
    echo "========================================"
    echo "CONFIG ATUAL: $DEVICE | $LABEL_STUDIO_URL"
    echo "========================================"
    echo "1. Iniciar servidores"
    echo "2. Auto-labeling interativo"
    echo "3. Parar servidores"
    echo "4. Resetar configuracoes (.env)"
    echo "5. Sair"
    echo "========================================"
    read -p "Escolha uma opcao (1-5): " choice

    case $choice in
        1) start_servers ;;
        2) auto_label ;;
        3) stop_servers ;;
        4) reset_env ;;
        5) exit 0 ;;
        *) echo "Opcao invalida." && sleep 1 ;;
    esac
done
