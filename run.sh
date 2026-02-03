#!/bin/bash

# Verifique se o tmux está instalado
if ! command -v tmux &> /dev/null; then
    echo "tmux não está instalado. Instalando..."
    sudo apt-get update && sudo apt-get install tmux -y
fi

# Nome da sessão tmux
SESSION_NAME="label_studio_sam"

# Função para exibir menu
show_menu() {
    echo -e "\n========================================"
    echo "       Label Studio + SAM2 Menu"
    echo "========================================"
    echo "1. Iniciar servidores"
    echo "2. Auto-labeling interativo"
    echo "3. Parar servidores"
    echo "4. Sair"
    echo "========================================"
}

# Função para verificar sessão
is_session_active() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Carregar .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Variáveis de ambiente carregadas."
else
    echo "Aviso: Arquivo .env não encontrado."
fi

while true; do
    show_menu
    read -p "Escolha uma opção (1-4): " choice

    case $choice in
        1)
            if is_session_active; then
                echo "Sessão já ativa. Abrindo novo terminal..."
                gnome-terminal -- bash -c "tmux attach -t $SESSION_NAME" &
            else
                echo "Criando nova sessão e abrindo terminal..."
                # Cria a sessão em background
                tmux new-session -d -s "$SESSION_NAME"
                
                # Setup Backend
                tmux send-keys -t "$SESSION_NAME" "source ~/Documentos/SAMVenv/bin/activate" C-m
                tmux send-keys -t "$SESSION_NAME" "cd ~/Documentos/labelStudio/label-studio-ml-backend/label_studio_ml/examples/" C-m
                tmux send-keys -t "$SESSION_NAME" "export DEVICE='cpu' LABEL_STUDIO_URL='http://127.0.0.1:8080'" C-m
                [ -n "$API_KEY" ] && tmux send-keys -t "$SESSION_NAME" "export LABEL_STUDIO_API_KEY='$API_KEY'" C-m
                tmux send-keys -t "$SESSION_NAME" "label-studio-ml start ./segment_anything_2_image" C-m
                
                # Setup Frontend
                tmux split-window -v -t "$SESSION_NAME"
                tmux send-keys -t "$SESSION_NAME" "source ~/Documentos/labelStudio/labelStudioVenv/bin/activate" C-m
                tmux send-keys -t "$SESSION_NAME" "label-studio start" C-m
                
                # Abre o terminal físico chamando o tmux
                gnome-terminal -- bash -c "tmux attach -t $SESSION_NAME" &
            fi
            ;;
        2)
            if ! is_session_active; then
                echo "Erro: Inicie os servidores primeiro."
            else
                # Se não houver API_KEY, solicita
                if [ -z "$API_KEY" ]; then
                    read -sp "Insira seu Token do Label Studio: " USER_TOKEN
                    echo ""
                    [ -z "$USER_TOKEN" ] && continue
                    export LABEL_STUDIO_API_KEY="$USER_TOKEN"
                else
                    export LABEL_STUDIO_API_KEY="$API_KEY"
                fi
                
                source ~/Documentos/SAMVenv/bin/activate
                cd ~/Documentos/labelStudio/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image
                export LABEL_STUDIO_URL='http://127.0.0.1:8080' DEVICE='cpu'
                python3 auto_label_cli.py
            fi
            ;;
        3)
            if is_session_active; then
                tmux kill-session -t "$SESSION_NAME"
                echo "Servidores finalizados."
            else
                echo "Nenhuma sessão ativa."
            fi
            ;;
        4)
            echo "Saindo..."
            exit 0
            ;;
        *)
            echo "Opção inválida."
            ;;
    esac
done
