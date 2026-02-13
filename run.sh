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
    unset DEVICE LABEL_STUDIO_URL LABEL_STUDIO_ML_BACKEND_URL PERSONAL_TOKEN LEGACY_TOKEN MODEL_CHECKPOINT MODEL_CONFIG YOLO_PLATE_MODEL_PATH YOLO_VEHICLE_MODEL_PATH SELECTED_BACKEND MODEL_DIR DEBUG_DUMP
}

init_config() {
    [ ! -f "$ENV_FILE" ] && touch "$ENV_FILE"
    load_env

    # [cite_start]Configuração dos caminhos padrão dos modelos YOLO [cite: 21]
    if [ -z "$YOLO_PLATE_MODEL_PATH" ]; then
        YOLO_PLATE_MODEL_PATH="label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/models/best.pt"
    fi
    if [ -z "$YOLO_VEHICLE_MODEL_PATH" ]; then
        YOLO_VEHICLE_MODEL_PATH="label-studio-ml-backend/label_studio_ml/examples/yolov11/models/yolo11x.pt"
    fi

    # [cite_start]Configuração do Dispositivo [cite: 22]
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

    # Configuração da URL do Frontend
    if [ -z "$LABEL_STUDIO_URL" ]; then
        echo
        read -p "[CONFIGURACAO] URL do Label Studio (Enter para padrao: http://localhost:8080): " IN_URL
        LABEL_STUDIO_URL=${IN_URL:-"http://localhost:8080"}
        echo "LABEL_STUDIO_URL=$LABEL_STUDIO_URL" >> "$ENV_FILE"
    fi

    # [cite_start]Configuração da URL do Backend ML [cite: 23]
    if [ -z "$LABEL_STUDIO_ML_BACKEND_URL" ]; then
        echo
        read -p "[CONFIGURACAO] URL do Backend ML (Enter para padrao: http://localhost:9090): " IN_ML_URL
        LABEL_STUDIO_ML_BACKEND_URL=${IN_ML_URL:-"http://localhost:9090"}
        echo "LABEL_STUDIO_ML_BACKEND_URL=$LABEL_STUDIO_ML_BACKEND_URL" >> "$ENV_FILE"
    fi
}

verify_yolo_models() {
    echo "Verificando modelos YOLO..."
    # [cite_start]Verificação do detector de placas [cite: 32, 37]
    if [ -f "$YOLO_PLATE_MODEL_PATH" ]; then
        echo "[SUCESSO] YOLO Plate detector encontrado."
    else
        echo "[AVISO] YOLO Plate detector nao encontrado. Baixando..."
        pushd "./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate" > /dev/null
        python3 download_model.py
        popd > /dev/null
    fi

    # [cite_start]Verificação do detector de veículos [cite: 33, 38]
    if [ -f "$YOLO_VEHICLE_MODEL_PATH" ]; then
        echo "[SUCESSO] YOLO Vehicle detector encontrado."
    else
        echo "[AVISO] YOLO Vehicle detector nao encontrado. Baixando..."
        pushd "./label-studio-ml-backend/label_studio_ml/examples/yolov11" > /dev/null
        python3 download_model.py
        popd > /dev/null
    fi
}

start_servers() {
    echo
    echo "[1] Iniciar Tudo (Frontend + Backend)"
    echo "[2] Apenas Frontend (Label Studio)"
    echo "[3] Apenas Backend ($SELECTED_BACKEND)"
    read -p "Escolha uma opcao: " OP_START

    # [cite_start]Define a pasta do backend [cite: 25]
    B_DIR="segment_anything_2_image"
    if [ "$SELECTED_BACKEND" = "YOLO" ]; then B_DIR="yolov11-plate"; fi

    BACKEND_PATH="./label-studio-ml-backend/label_studio_ml/examples/$B_DIR"

    if [ "$OP_START" = "1" ]; then
        if [ ! -d "$BACKEND_PATH" ]; then
            echo "[ERRO] Pasta $BACKEND_PATH nao encontrada."
            read -p "Pressione Enter para voltar..."
            return
        fi
        echo "Iniciando Frontend e Backend..."
        label-studio &
        sleep 2
        pushd "$BACKEND_PATH" > /dev/null
        # [cite_start]Exporta variáveis para o backend [cite: 27]
        export DEVICE=$DEVICE
        export LABEL_STUDIO_URL=$LABEL_STUDIO_URL
        export LABEL_STUDIO_API_KEY=$PERSONAL_TOKEN
        label-studio-ml start . &
        popd > /dev/null
    fi

    if [ "$OP_START" = "2" ]; then
        label-studio &
    fi

    if [ "$OP_START" = "3" ]; then
        pushd "$BACKEND_PATH" > /dev/null
        export DEVICE=$DEVICE
        export LABEL_STUDIO_URL=$LABEL_STUDIO_URL
        export LABEL_STUDIO_API_KEY=$PERSONAL_TOKEN
        label-studio-ml start . &
        popd > /dev/null
    fi
    
    echo "Servidores disparados."
    sleep 2
}

auto_label() {
    load_env
    if [ -z "$SELECTED_BACKEND" ]; then
        echo "[ERRO] Configure o backend primeiro (Opcao 4)."
        read -p "Pressione Enter para continuar..."
        return
    fi

    # --- TOKENS ---
    if [ -z "$PERSONAL_TOKEN" ]; then
        read -p "[CONFIGURACAO] Informe o PERSONAL_TOKEN do Label Studio: " IN_PT
        [ ! -z "$IN_PT" ] && PERSONAL_TOKEN=$IN_PT && echo "PERSONAL_TOKEN=$PERSONAL_TOKEN" >> "$ENV_FILE"
    fi
    export LABEL_STUDIO_API_KEY=$PERSONAL_TOKEN

    if [ "$SELECTED_BACKEND" = "YOLO" ]; then
        verify_yolo_models
        echo "Iniciando Auto-labeling CLI (YOLOv11)..."
        source ./labelStudioVenv/bin/activate
        ROOT_DIR="$(pwd)"
        pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate > /dev/null
        python3 auto_label_cli.py \
            --model_path="$ROOT_DIR/$YOLO_PLATE_MODEL_PATH" \
            --vehicle_model_path="$ROOT_DIR/$YOLO_VEHICLE_MODEL_PATH"
        popd > /dev/null
    else
        echo "Iniciando Auto-labeling CLI (SAM2)..."
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
    echo "Processos encerrados."
    sleep 1
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
    echo "=========================================================="
    echo "      Label Studio Auto-Labeling Tool (Linux)"
    echo "=========================================================="
    echo "CONFIG ATUAL  : $DEVICE"
    echo "CONFIG FRONTEND: $LABEL_STUDIO_URL"
    echo "CONFIG BACKEND : $LABEL_STUDIO_ML_BACKEND_URL"
    echo "BACKEND PADRAO : $SELECTED_BACKEND"
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