#!/bin/bash

# --- CONFIGURACAO ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$BASE_DIR/.env"
PYTHON="$BASE_DIR/labelStudioVenv/bin/python"

# Função para carregar variáveis do .env com suporte a espaços simples
load_env() {
    if [ -f "$ENV_FILE" ]; then
        # Exporta variáveis ignorando comentários e linhas vazias
        export $(grep -v '^#' "$ENV_FILE" | xargs -d '\n')
    fi
}

# Função para limpar memória (variáveis da sessão)
reset_memory() {
    unset LABEL_STUDIO_URL PERSONAL_TOKEN LEGACY_TOKEN YOLO_ENABLE_OPENVINO YOLO_PLATE_MODEL_PATH YOLO_VEHICLE_MODEL_PATH
}

resolve_path() {
    local p="$1"
    if [ -z "$p" ]; then
        echo ""
        return 0
    fi
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$BASE_DIR/$p"
    fi
}

start_labelstudio_terminal() {
    local cmd="cd \"$BASE_DIR\" && source ./labelStudioVenv/bin/activate && label-studio start"

    if command -v konsole >/dev/null 2>&1; then
        konsole --noclose -e bash -lc "$cmd" &
        return 0
    fi
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal -- bash -lc "$cmd; exec bash" &
        return 0
    fi
    if command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --hold -e "bash -lc '$cmd'" &
        return 0
    fi
    if command -v xterm >/dev/null 2>&1; then
        xterm -hold -e bash -lc "$cmd" &
        return 0
    fi

    return 1
}

ensure_venv() {
    if [ ! -x "$PYTHON" ]; then
        echo
        echo "[ERRO] Ambiente virtual nao encontrado em ./labelStudioVenv."
        while true; do
            read -p "Deseja executar o setup agora? [S/n]: " IN_SETUP
            [ -z "$IN_SETUP" ] && IN_SETUP="S"
            if [ "$IN_SETUP" = "S" ] || [ "$IN_SETUP" = "s" ]; then
                (cd "$BASE_DIR" && ./setup.sh)
                if [ $? -ne 0 ]; then
                    echo "[ERRO] Falha ao executar setup.sh."
                    return 1
                fi
                break
            fi
            if [ "$IN_SETUP" = "N" ] || [ "$IN_SETUP" = "n" ]; then
                echo "[AVISO] Nao sera possivel continuar sem a venv."
                return 1
            fi
            echo "[AVISO] Opcao invalida."
        done
    fi

    if [ ! -x "$PYTHON" ]; then
        echo "[ERRO] Python da venv nao encontrado em $PYTHON."
        return 1
    fi
}

init_config() {
    [ ! -f "$ENV_FILE" ] && touch "$ENV_FILE"
    load_env

    if [ -z "$YOLO_PLATE_MODEL_PATH" ]; then
        echo
        DEFAULT_PLATE="label-studio-ml-backend/label_studio_ml/examples/yolov11-plate/models/best.pt"
        read -p "[CONFIGURACAO] Caminho do modelo yolo11-plate (Enter para padrao): " IN_PP
        YOLO_PLATE_MODEL_PATH=${IN_PP:-"$DEFAULT_PLATE"}
        echo "YOLO_PLATE_MODEL_PATH=$YOLO_PLATE_MODEL_PATH" >> "$ENV_FILE"
    fi

    if [ -z "$YOLO_VEHICLE_MODEL_PATH" ]; then
        echo
        DEFAULT_VEHICLE="label-studio-ml-backend/label_studio_ml/examples/yolov11/models/yolo11x.pt"
        read -p "[CONFIGURACAO] Caminho do modelo yolo11x (Enter para padrao): " IN_VP
        YOLO_VEHICLE_MODEL_PATH=${IN_VP:-"$DEFAULT_VEHICLE"}
        echo "YOLO_VEHICLE_MODEL_PATH=$YOLO_VEHICLE_MODEL_PATH" >> "$ENV_FILE"
    fi

    if [ -z "$LABEL_STUDIO_URL" ]; then
        echo
        read -p "[CONFIGURACAO] URL do Label Studio (Enter para padrao: http://localhost:8080): " IN_URL
        LABEL_STUDIO_URL=${IN_URL:-"http://localhost:8080"}
        echo "LABEL_STUDIO_URL=$LABEL_STUDIO_URL" >> "$ENV_FILE"
    fi
}

verify_yolo_models() {
    ensure_venv || return 1
    if [ -z "$YOLO_ENABLE_OPENVINO" ]; then
        while true; do
            echo
            read -p "[CONFIGURACAO] Deseja ativar execucao com openVINO? [S/n]: " IN_OV
            if [ -z "$IN_OV" ] || [ "$IN_OV" = "S" ] || [ "$IN_OV" = "s" ]; then
                YOLO_ENABLE_OPENVINO=1
                break
            fi
            if [ "$IN_OV" = "N" ] || [ "$IN_OV" = "n" ]; then
                YOLO_ENABLE_OPENVINO=0
                break
            fi
            echo "[AVISO] Opcao invalida."
        done
        echo "YOLO_ENABLE_OPENVINO=$YOLO_ENABLE_OPENVINO" >> "$ENV_FILE"
    fi

    echo
    echo "[DEBUG] Verificando arquivos dos modelos..."
    local plate_path
    plate_path="$(resolve_path "$YOLO_PLATE_MODEL_PATH")"
    if [ -f "$plate_path" ]; then
        echo "[DEBUG] Modelo yolo11-plate encontrado."
    else
        echo "[AVISO] Modelo yolo11-plate NAO encontrado."
        while true; do
            read -p "Deseja baixar o modelo yolo11-plate agora? [S/n]: " DL_P
            [ -z "$DL_P" ] && DL_P="S"
            if [ "$DL_P" = "S" ] || [ "$DL_P" = "s" ]; then
                pushd "$BASE_DIR/label-studio-ml-backend/label_studio_ml/examples/yolov11-plate" > /dev/null
                "$PYTHON" utils.py download_model
                popd > /dev/null
                break
            fi
            if [ "$DL_P" = "N" ] || [ "$DL_P" = "n" ]; then
                echo "[AVISO] Continuando sem baixar. Erros podem ocorrer na predicao."
                break
            fi
        done
    fi

    local vehicle_path
    vehicle_path="$(resolve_path "$YOLO_VEHICLE_MODEL_PATH")"
    if [ -f "$vehicle_path" ]; then
        echo "[DEBUG] Modelo yolo11x encontrado."
    else
        echo "[AVISO] Modelo yolo11x NAO encontrado."
        while true; do
            read -p "Deseja baixar o modelo yolo11x agora? [S/n]: " DL_V
            [ -z "$DL_V" ] && DL_V="S"
            if [ "$DL_V" = "S" ] || [ "$DL_V" = "s" ]; then
                pushd "$BASE_DIR/label-studio-ml-backend/label_studio_ml/examples/yolov11" > /dev/null
                "$PYTHON" utils.py download_model
                popd > /dev/null
                break
            fi
            if [ "$DL_V" = "N" ] || [ "$DL_V" = "n" ]; then
                echo "[AVISO] Continuando sem baixar."
                break
            fi
        done
    fi
}

start_server() {
    ensure_venv || return 1
    echo
    echo "Iniciando Frontend (Label Studio)..."
    if start_labelstudio_terminal; then
        echo "Frontend iniciado em $LABEL_STUDIO_URL (novo terminal)."
    else
        echo "[AVISO] Nenhum terminal grafico encontrado. Iniciando no processo atual."
        source "$BASE_DIR/labelStudioVenv/bin/activate"
        label-studio start &
        echo "Frontend iniciado em $LABEL_STUDIO_URL"
    fi
    echo
    read -p "Pressione Enter para continuar..."
}

auto_label() {
    load_env
    ensure_venv || return 1

    if [ -z "$LEGACY_TOKEN" ]; then
        while true; do
            echo
            read -p "[CONFIGURACAO] Informe o LEGACY TOKEN do Label Studio: " IN_LT
            if [ -n "$IN_LT" ]; then
                LEGACY_TOKEN=$IN_LT
                echo "LEGACY_TOKEN=$LEGACY_TOKEN" >> "$ENV_FILE"
                break
            fi
            echo "[ERRO] O token nao pode estar vazio."
        done
    fi

    if [ -z "$PERSONAL_TOKEN" ]; then
        while true; do
            echo
            read -p "[CONFIGURACAO] Informe o PERSONAL TOKEN do Label Studio: " IN_PT
            if [ -n "$IN_PT" ]; then
                PERSONAL_TOKEN=$IN_PT
                echo "PERSONAL_TOKEN=$PERSONAL_TOKEN" >> "$ENV_FILE"
                break
            fi
            echo "[ERRO] O token nao pode estar vazio."
        done
    fi

    verify_yolo_models || return 1
    echo
    echo "Iniciando Auto-labeling CLI (YOLOv11)..."
    source "$BASE_DIR/labelStudioVenv/bin/activate"
    export LABEL_STUDIO_API_KEY="$PERSONAL_TOKEN"
    export PERSONAL_TOKEN="$PERSONAL_TOKEN"
    export LEGACY_TOKEN="$LEGACY_TOKEN"
    ROOT_DIR="$BASE_DIR"
    pushd "$BASE_DIR/label-studio-ml-backend/label_studio_ml/examples/yolov11-plate" > /dev/null
    "$PYTHON" auto_label_cli.py \
        --model_path="$(resolve_path "$YOLO_PLATE_MODEL_PATH")" \
        --vehicle_model_path="$(resolve_path "$YOLO_VEHICLE_MODEL_PATH")"
    popd > /dev/null
    echo
    read -p "Pressione Enter para continuar..."
}

stop_servers() {
    echo "Parando servidores..."
    pkill -f "label-studio"
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
    echo "CONFIG FRONTEND: $LABEL_STUDIO_URL"
    echo "========================================"
    echo "1. Iniciar servidor"
    echo "2. Auto-labeling interativo (YOLO)"
    echo "3. Parar servidor"
    echo "4. Resetar configuracoes (.env)"
    echo "5. Sair"
    echo "========================================"
    read -p "Escolha uma opcao: " choice

    case $choice in
        1) start_server ;;
        2) auto_label ;;
        3) stop_servers ;;
        4) reset_env ;;
        5) exit 0 ;;
        *) echo "Opcao invalida." && sleep 1 ;;
    esac
done
