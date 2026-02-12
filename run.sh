#!/bin/bash

# --- CONFIGURAÇÃO ---
ENV_FILE=".env"

# Função para carregar variáveis do .env com suporte a espaços simples
load_env() {
    if [ -f "$ENV_FILE" ]; then
        # Exporta variáveis ignorando comentários e linhas vazias
        export $(grep -v '^#' "$ENV_FILE" | xargs -d '\n')
    fi
}

# Função para limpar memória (variáveis da sessão)
reset_memory() {
    unset DEVICE LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN MODEL_CHECKPOINT MODEL_CONFIG YOLO_PLATE_MODEL_PATH YOLO_VEHICLE_MODEL_PATH SELECTED_BACKEND MODEL_DIR DEBUG_DUMP
}

init_config() {
    [ ! -f "$ENV_FILE" ] && touch "$ENV_FILE"
    load_env

    # Configuração do Dispositivo
    if [ -z "$DEVICE" ]; then
        echo
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
        echo
        read -p "[CONFIGURACAO] URL do Label Studio (Padrao: http://127.0.0.1:8080): " IN_URL
        LABEL_STUDIO_URL=${IN_URL:-"http://127.0.0.1:8080"}
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

    export BASE_DIR DEVICE LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN

    gnome-terminal -- bash -c "
        cd \"$BASE_DIR\" || exit 1
        source ./labelStudioVenv/bin/activate

        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            tmux attach -t $SESSION_NAME
            exec bash
        fi

        tmux new-session -d -s $SESSION_NAME
        tmux split-window -h -t $SESSION_NAME

        # Pane 0 - Frontend
        tmux send-keys -t $SESSION_NAME:0.0 \"label-studio start\" C-m

        # Pane 1 - Backend
        tmux send-keys -t $SESSION_NAME:0.1 \"
            export DEVICE=$DEVICE
            export LABEL_STUDIO_URL=$LABEL_STUDIO_URL
            cd ./label-studio-ml-backend/label_studio_ml/examples/
            label-studio-ml start ./segment_anything_2_image
        \" C-m

        tmux attach -t $SESSION_NAME
        exec bash
    "
}

auto_label() {
    load_env
    
    # --- TOKENS (Garantindo export para o Python) ---
    if [ -z "$PERSONAL_TOKEN" ]; then
        read -p "[CONFIGURACAO] Informe o PERSONAL_TOKEN do Label Studio: " IN_PT
        [ ! -z "$IN_PT" ] && PERSONAL_TOKEN=$IN_PT && echo "PERSONAL_TOKEN=$PERSONAL_TOKEN" >> "$ENV_FILE"
    fi
    export PERSONAL_TOKEN=$PERSONAL_TOKEN
    export LABEL_STUDIO_API_KEY=$PERSONAL_TOKEN

    if [ -z "$LEGACY_TOKEN" ]; then
        read -p "[CONFIGURACAO] Informe o LEGACY_TOKEN do Label Studio: " IN_LT
        [ ! -z "$IN_LT" ] && LEGACY_TOKEN=$IN_LT && echo "LEGACY_TOKEN=$LEGACY_TOKEN" >> "$ENV_FILE"
    fi
    export LEGACY_TOKEN=$LEGACY_TOKEN

    # Debug Dump config
    if [ -z "$DEBUG_DUMP" ]; then
        read -p "[CONFIGURACAO] Habilitar debug dumps (json/png)? (S/n): " IN_DEBUG
        if [ "$IN_DEBUG" = "n" ] || [ "$IN_DEBUG" = "N" ]; then
            DEBUG_DUMP=0
        else
            DEBUG_DUMP=1
        fi
        echo "DEBUG_DUMP=$DEBUG_DUMP" >> "$ENV_FILE"
    fi
    export DEBUG_DUMP

    # Seleção de Backend
    if [ -z "$SELECTED_BACKEND" ]; then
        echo
        read -p "[CONFIGURACAO] Escolha backend: 1) SAM2  2) YOLOv11 (Enter = 1): " IN_BACK
        if [ "$IN_BACK" = "2" ]; then
            SELECTED_BACKEND="YOLO"
        else
            SELECTED_BACKEND="SAM2"
        fi
        echo "SELECTED_BACKEND=$SELECTED_BACKEND" >> "$ENV_FILE"
    fi

    export DEVICE=$DEVICE
    export LABEL_STUDIO_URL=$LABEL_STUDIO_URL

    if [ "$SELECTED_BACKEND" = "YOLO" ]; then
        # --- MODEL_DIR ---
        if [ -z "$MODEL_DIR" ]; then
            DEFAULT_MODEL_DIR="$(pwd)/model_runs"
            read -p "[CONFIGURACAO] Diretorio para logs/modelos (Enter para padrao): " IN_MD
            MODEL_DIR=${IN_MD:-"$DEFAULT_MODEL_DIR"}
            echo "MODEL_DIR=$MODEL_DIR" >> "$ENV_FILE"
        fi
        mkdir -p "$MODEL_DIR"
        export MODEL_DIR

        # --- CONFIGURAÇÃO YOLO PATHS ---
        YOLO_PLATE_MODEL_PATH=${YOLO_PLATE_MODEL_PATH:-"label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/models/best.pt"}
        YOLO_VEHICLE_MODEL_PATH=${YOLO_VEHICLE_MODEL_PATH:-"label-studio-ml-backend/label_studio_ml/examples/yolov11/models/yolo11x.pt"}

        # --- CORREÇÃO DO DOWNLOAD (Entrando na pasta antes de executar) ---
        if [ ! -f "$YOLO_PLATE_MODEL_PATH" ]; then
            echo "[AVISO] YOLO Plate detector não encontrado. Baixando..."
            pushd "./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/" > /dev/null
            python3 download_model.py
            popd > /dev/null
        fi

        if [ ! -f "$YOLO_VEHICLE_MODEL_PATH" ]; then
            echo "[AVISO] YOLO Vehicle detector não encontrado. Baixando..."
            pushd "./label-studio-ml-backend/label_studio_ml/examples/yolov11/" > /dev/null
            python3 download_model.py
            popd > /dev/null
        fi

        echo "Iniciando Auto-labeling CLI (YOLOv11)..."
        source ./labelStudioVenv/bin/activate
        
        # Salva o diretório raiz antes do pushd
        ROOT_DIR="$(pwd)"
        
        pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate > /dev/null
        # Executa usando caminhos absolutos baseados na raiz do projeto
        python3 auto_label_cli.py \
            --model_path="$ROOT_DIR/$YOLO_PLATE_MODEL_PATH" \
            --vehicle_model_path="$ROOT_DIR/$YOLO_VEHICLE_MODEL_PATH"
        popd > /dev/null
    else
        # Fluxo SAM2
        source ./labelStudioVenv/bin/activate
        pushd ./label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image > /dev/null
        python3 auto_label_cli.py
        popd > /dev/null
    fi
    
    read -p "Pressione Enter para continuar..."
}

stop_servers() {
    echo "Parando servidores..."
    pkill -f "label-studio"
    pkill -f "label-studio-ml"
    tmux kill-session -t labelstudio 2>/dev/null
    echo "Processos encerrados."
    sleep 2
}

reset_env() {
    echo "Apagando configuracoes e limpando memoria..."
    [ -f "$ENV_FILE" ] && rm "$ENV_FILE"
    reset_memory
}

# Loop do Menu Principal
while true; do
    init_config
    clear
    echo "========================================"
    echo "      Label Studio + SAM2 Menu (Linux)"
    echo "========================================"
    echo "CONFIG ATUAL: $DEVICE | $LABEL_STUDIO_URL"
    echo "BACKEND PADRAO: $SELECTED_BACKEND"
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