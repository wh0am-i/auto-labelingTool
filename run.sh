#!/bin/bash

set -u

cd "$(dirname "$0")"

ENV_FILE="2.env"
PYTHON="./labelStudioVenv/bin/python"

# --- LIMPEZA DE MEMORIA DA SESSAO ---
unset LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN YOLO_ENABLE_OPENVINO YOLO_FORCE_OPENVINO YOLO_PLATE_MODEL_PATH YOLO_VEHICLE_MODEL_PATH

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf "%s" "$s"
}

load_env() {
    [ -f "$ENV_FILE" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="$(trim "$line")"
        [ -z "$line" ] && continue
        [[ "$line" == \#* ]] && continue
        key="${line%%=*}"
        val="${line#*=}"
        key="$(trim "$key")"
        val="$(trim "$val")"
        [ -z "$key" ] && continue
        export "$key=$val"
    done < "$ENV_FILE"
}

init_config() {
    [ -f "$ENV_FILE" ] || touch "$ENV_FILE"
    load_env

    if [ -z "${YOLO_PLATE_MODEL_PATH:-}" ]; then
        echo
        DEFAULT_PLATE="label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/models/best.pt"
        read -r -p "[CONFIGURACAO] Caminho do modelo yolo11-plate (Enter para padrao): " IN_PP
        if [ -z "$IN_PP" ]; then
            YOLO_PLATE_MODEL_PATH="$DEFAULT_PLATE"
        else
            YOLO_PLATE_MODEL_PATH="$IN_PP"
        fi
        echo "YOLO_PLATE_MODEL_PATH=$YOLO_PLATE_MODEL_PATH" >> "$ENV_FILE"
    fi

    if [ -z "${YOLO_VEHICLE_MODEL_PATH:-}" ]; then
        echo
        DEFAULT_VEHICLE="label-studio-ml-backend/label_studio_ml/examples/yolov11/models/yolo11x.pt"
        read -r -p "[CONFIGURACAO] Caminho do modelo yolo11x (Enter para padrao): " IN_VP
        if [ -z "$IN_VP" ]; then
            YOLO_VEHICLE_MODEL_PATH="$DEFAULT_VEHICLE"
        else
            YOLO_VEHICLE_MODEL_PATH="$IN_VP"
        fi
        echo "YOLO_VEHICLE_MODEL_PATH=$YOLO_VEHICLE_MODEL_PATH" >> "$ENV_FILE"
    fi

    if [ -z "${LABEL_STUDIO_URL:-}" ]; then
        echo
        read -r -p "[CONFIGURACAO] URL do Label Studio (Enter para padrao: http://localhost:8080): " IN_URL
        LABEL_STUDIO_URL="${IN_URL:-http://localhost:8080}"
        echo "LABEL_STUDIO_URL=$LABEL_STUDIO_URL" >> "$ENV_FILE"
    fi
}

start_server() {
    echo
    echo "Iniciando Frontend (Label Studio)..."
    source ./labelStudioVenv/bin/activate
    label-studio start &
    sleep 2
    echo "Frontend iniciado em $LABEL_STUDIO_URL"
    echo
    read -r -p "Pressione Enter para voltar..."
}

verify_yolo_models() {
    if [ -z "${YOLO_ENABLE_OPENVINO:-}" ]; then
        while true; do
            echo
            read -r -p "[CONFIGURACAO] Deseja ativar execucao com openVINO? [S/n]: " IN_OV
            if [ -z "$IN_OV" ] || [[ "$IN_OV" =~ ^[Ss]$ ]]; then
                YOLO_ENABLE_OPENVINO=1
                YOLO_FORCE_OPENVINO=1
                break
            elif [[ "$IN_OV" =~ ^[Nn]$ ]]; then
                YOLO_ENABLE_OPENVINO=0
                YOLO_FORCE_OPENVINO=0
                break
            else
                echo "[AVISO] Opcao invalida."
            fi
        done
        echo "YOLO_ENABLE_OPENVINO=$YOLO_ENABLE_OPENVINO" >> "$ENV_FILE"
        echo "YOLO_FORCE_OPENVINO=$YOLO_FORCE_OPENVINO" >> "$ENV_FILE"
    fi

    echo
    echo "[DEBUG] Verificando arquivos dos modelos..."

    if [ -f "$YOLO_PLATE_MODEL_PATH" ]; then
        echo "[DEBUG] Modelo yolo11-plate encontrado."
    else
        echo "[AVISO] Modelo yolo11-plate NAO encontrado."
        while true; do
            read -r -p "Deseja baixar o modelo yolo11-plate agora? [S/n]: " DL_P
            DL_P="${DL_P:-S}"
            if [[ "$DL_P" =~ ^[Ss]$ ]]; then
                pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate > /dev/null
                "$PYTHON" download_model.py
                popd > /dev/null
                break
            elif [[ "$DL_P" =~ ^[Nn]$ ]]; then
                echo "[AVISO] Continuando sem baixar. Erros podem ocorrer na predicao."
                break
            else
                echo "[AVISO] Opcao invalida."
            fi
        done
    fi

    if [ -f "$YOLO_VEHICLE_MODEL_PATH" ]; then
        echo "[DEBUG] Modelo yolo11x encontrado."
    else
        echo "[AVISO] Modelo yolo11x NAO encontrado."
        while true; do
            read -r -p "Deseja baixar o modelo yolo11x agora? [S/n]: " DL_V
            DL_V="${DL_V:-S}"
            if [[ "$DL_V" =~ ^[Ss]$ ]]; then
                pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11 > /dev/null
                "$PYTHON" download_model.py
                popd > /dev/null
                break
            elif [[ "$DL_V" =~ ^[Nn]$ ]]; then
                echo "[AVISO] Continuando sem baixar."
                break
            else
                echo "[AVISO] Opcao invalida."
            fi
        done
    fi
}

auto_label() {
    if [ -z "${LEGACY_TOKEN:-}" ]; then
        while true; do
            echo
            read -r -p "[CONFIGURACAO] Informe o LEGACY TOKEN do Label Studio: " TOKEN
            if [ -n "$TOKEN" ]; then
                LEGACY_TOKEN="$TOKEN"
                echo "LEGACY_TOKEN=$LEGACY_TOKEN" >> "$ENV_FILE"
                break
            fi
            echo "[ERRO] O token nao pode estar vazio."
        done
    fi

    if [ -z "${PERSONAL_TOKEN:-}" ]; then
        while true; do
            echo
            read -r -p "[CONFIGURACAO] Informe o PERSONAL TOKEN do Label Studio: " TOKEN
            if [ -n "$TOKEN" ]; then
                PERSONAL_TOKEN="$TOKEN"
                echo "PERSONAL_TOKEN=$PERSONAL_TOKEN" >> "$ENV_FILE"
                break
            fi
            echo "[ERRO] O token nao pode estar vazio."
        done
    fi

    verify_yolo_models

    echo
    echo "Iniciando Auto-labeling CLI (YOLOv11)..."
    source ./labelStudioVenv/bin/activate
    export LABEL_STUDIO_API_KEY="$PERSONAL_TOKEN"
    ROOT_DIR="$(pwd)"
    pushd ./label-studio-ml-backend/label_studio_ml/examples/yolov11-plate > /dev/null
    "$PYTHON" auto_label_cli.py \
        --model_path="$ROOT_DIR/$YOLO_PLATE_MODEL_PATH" \
        --vehicle_model_path="$ROOT_DIR/$YOLO_VEHICLE_MODEL_PATH"
    popd > /dev/null
    echo
    read -r -p "Pressione Enter para voltar..."
}

stop_servers() {
    echo
    echo "Encerrando processos do Label Studio..."
    pkill -f "label-studio start" >/dev/null 2>&1 || true
    pkill -f "label-studio" >/dev/null 2>&1 || true
    echo "Concluido."
    echo
    read -r -p "Pressione Enter para voltar..."
}

reset_env() {
    echo "Apagando configuracoes e limpando memoria..."
    [ -f "$ENV_FILE" ] && rm -f "$ENV_FILE"
    unset LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN YOLO_ENABLE_OPENVINO YOLO_FORCE_OPENVINO YOLO_PLATE_MODEL_PATH YOLO_VEHICLE_MODEL_PATH
    init_config
}

while true; do
    init_config
    clear
    echo "=========================================================="
    echo "      Label Studio Auto-Labeling Tool (Linux)"
    echo "=========================================================="
    echo "CONFIG FRONTEND: ${LABEL_STUDIO_URL:-}"
    echo "========================================"
    echo "1. Iniciar servidor"
    echo "2. Auto-labeling interativo (YOLO)"
    echo "3. Parar servidor"
    echo "4. Resetar configuracoes (2.env)"
    echo "5. Sair"
    echo "========================================"
    read -r -p "Escolha uma opcao: " choice

    case "$choice" in
        1) start_server ;;
        2) auto_label ;;
        3) stop_servers ;;
        4) reset_env ;;
        5) exit 0 ;;
        *) echo "Opcao invalida."; sleep 1 ;;
    esac
done
