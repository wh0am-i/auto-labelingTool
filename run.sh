#!/bin/bash

# Verifique se o tmux está instalado
if ! command -v tmux &> /dev/null
then
    echo "tmux não está instalado. Instalando."
    sudo apt-get install tmux
    exit 1
fi

# Nome da sessão tmux
SESSION_NAME="my_tmux_session"

# Função para exibir menu
show_menu() {
    echo ""
    echo "========================================"
    echo "Label Studio + SAM2 Menu"
    echo "========================================"
    echo "1. Iniciar servidores (backend + frontend)"
    echo "2. Auto-labeling interativo (selecionar projeto)"
    echo "3. Parar servidores"
    echo "4. Sair"
    echo "========================================"
    echo ""
}

# Função para verificar se a sessão está realmente ativa
is_session_active() {
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        # Verifica se há janelas na sessão
        if [ $(tmux list-windows -t $SESSION_NAME 2>/dev/null | wc -l) -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

# Carregar variáveis de ambiente do arquivo .env (mais robusto)
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # ignora comentários e linhas vazias (começando com # ou //)
        trimmed=$(echo "$line" | sed -e 's/^[[:space:]]*//')
        case "$trimmed" in
            ''|\#*|\/\/*)
                continue
                ;;
            *)
                key=$(echo "$trimmed" | cut -d= -f1)
                val=$(echo "$trimmed" | cut -d= -f2-)
                export "$key=$val"
                ;;
        esac
    done < .env
    echo "Variáveis de ambiente carregadas do arquivo .env"
else
    echo "Arquivo .env não encontrado."
fi

# Menu principal
while true; do
    show_menu
    read -p "Escolha uma opção (1-4): " choice
    
    case $choice in
        1)
            if is_session_active; then
                echo "Sessão já está em execução."
                echo "Anexando à sessão existente..."
                sleep 1
                tmux attach -t $SESSION_NAME
            else
                echo "Iniciando servidores..."
                
                # Inicia uma nova sessão tmux
                tmux new-session -d -s $SESSION_NAME
                
                # Backend SAM2
                tmux send-keys -t $SESSION_NAME "source ~/Documentos/SAMVenv/bin/activate" C-m
                tmux send-keys -t $SESSION_NAME "echo 'Ambiente SAM ativado'" C-m
                tmux send-keys -t $SESSION_NAME "cd ~/Documentos/labelStudio/label-studio-ml-backend/label_studio_ml/examples/" C-m
                tmux send-keys -t $SESSION_NAME "export DEVICE='cpu'" C-m
                tmux send-keys -t $SESSION_NAME "export LABEL_STUDIO_URL='http://127.0.0.1:8080'" C-m
                # export API key into tmux backend session if available
                if [ -n "$API_KEY" ]; then
                    tmux send-keys -t $SESSION_NAME "export LABEL_STUDIO_API_KEY='$API_KEY'" C-m
                fi
                tmux send-keys -t $SESSION_NAME "label-studio-ml start ./segment_anything_2_image" C-m
                
                # Frontend Label Studio
                tmux split-window -t $SESSION_NAME -v
                tmux send-keys -t $SESSION_NAME "source ~/Documentos/labelStudio/labelstudio/bin/activate" C-m
                tmux send-keys -t $SESSION_NAME "echo 'Ambiente Label Studio ativado'" C-m
                tmux send-keys -t $SESSION_NAME "label-studio start" C-m
                
                echo "Servidores iniciados com sucesso!"
                echo "Backend SAM2: http://127.0.0.1:9090"
                echo "Frontend Label Studio: http://0.0.0.0:8080"
                echo "Aguarde 3 segundos para que os serviços iniciem..."
                sleep 3
                
                echo "Anexando à sessão..."
                tmux attach -t $SESSION_NAME
            fi
            ;;
        2)
            if ! is_session_active; then
                echo "Erro: Servidores não estão em execução. Inicie-os primeiro (opção 1)."
                sleep 2
            else
                echo "Abrindo auto-labeling interativo..."
                
                # Verifica se a API_KEY está configurado, se não, pede ao usuário ou informa que não está disponível
                if [ -z "$API_KEY" ]; then
                    echo ""
                    echo "Erro: API_KEY não encontrado no arquivo .env."
                    echo "Você pode definir a chave no arquivo .env ou inserir manualmente."
                    read -sp "Insira seu Label Studio Personal Access Token (ou ENTER para cancelar): " USER_TOKEN
                    echo ""
                    if [ -z "$USER_TOKEN" ]; then
                        echo "Saindo do modo Auto-labeling."
                        continue
                    fi
                    export API_KEY="$USER_TOKEN"
                fi
                
                # Se a chave estiver no .env ou informada, continua com o auto-labeling
                export LABEL_STUDIO_API_KEY="$API_KEY"
                
                # Executa o auto-labeling
                source ~/Documentos/SAMVenv/bin/activate
                cd ~/Documentos/labelStudio/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image
                export LABEL_STUDIO_URL='http://127.0.0.1:8080'
                export DEVICE='cpu'
                python auto_label_cli.py
            fi
            ;;
        3)
            if is_session_active; then
                echo "Parando servidores..."
                tmux kill-session -t $SESSION_NAME
                echo "Servidores parados."
                sleep 1
            else
                echo "Nenhuma sessão em execução."
                sleep 1
            fi
            ;;
        4)
            echo "Saindo..."
            exit 0
            ;;
        *)
            echo "Opção inválida. Tente novamente."
            sleep 1
            ;;
    esac
done