#!/usr/bin/env python3

import os
import sys
import logging
import time
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_label_studio_connection(url, max_retries=5):
    """Verifica se Label Studio está acessível"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Conectado ao Label Studio com sucesso!")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                logger.warning(f"Tentativa {attempt + 1}/{max_retries}: Aguardando Label Studio...")
                time.sleep(2)
            continue
    
    logger.error(f"✗ Não foi possível conectar ao Label Studio em {url}")
    return False


def check_sam2_checkpoint():
    """Verifica se o checkpoint SAM2 existe em várias localizações comuns"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoint_name = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')

    candidates = [
        os.path.join(script_dir, "checkpoints", checkpoint_name),        # plugin/checkpoints
        os.path.join(examples_dir, "checkpoints", checkpoint_name),      # examples/checkpoints
        os.path.join(examples_dir, checkpoint_name),                     # examples/<name>
        os.path.join(script_dir, "..", "checkpoints", checkpoint_name),  # parent check
        checkpoint_name,                                                 # cwd relative
        os.path.abspath(os.path.join(os.path.expanduser("~"), "Documentos", "segment-anything-2", "checkpoints", checkpoint_name)),
    ]

    for c in candidates:
        if c and os.path.exists(c):
            logger.info(f"✓ Checkpoint encontrado: {c}")
            return True

    logger.error(f"✗ Checkpoint SAM2 não encontrado. Procurado: {checkpoint_name}")
    logger.info("\n📥 Para baixar o checkpoint, execute:")
    logger.info("   cd ~/Documentos/segment-anything-2")
    logger.info("   python scripts/download_ckpts.py")
    logger.info("\nOu coloque o arquivo em uma das pastas:")
    for c in candidates:
        logger.info(" - %s", os.path.abspath(c))
    return False


def main():
    """Interface CLI para auto-labeling interativo"""
    
    # Configurações
    label_studio_url = os.getenv('LABEL_STUDIO_URL', 'http://127.0.0.1:8080')
    api_key = os.getenv('LABEL_STUDIO_API_KEY')
    
    # Valida API Key
    if not api_key:
        logger.error("✗ LABEL_STUDIO_API_KEY não configurada")
        logger.info("Use: export LABEL_STUDIO_API_KEY='seu_token'")
        sys.exit(1)
    
    # Valida conexão com Label Studio
    logger.info(f"Verificando conexão com Label Studio em {label_studio_url}...")
    if not check_label_studio_connection(label_studio_url):
        logger.error("Impossível conectar ao Label Studio. Verifique se está rodando.")
        sys.exit(1)
    
    # Valida checkpoint SAM2
    logger.info("Verificando checkpoint SAM2...")
    if not check_sam2_checkpoint():
        sys.exit(1)
    
    # Importa o modelo após validações
    try:
        from model import NewModel
    except ImportError as e:
        logger.error(f"Erro ao importar modelo: {e}")
        sys.exit(1)
    
    # Inicializa o modelo
    logger.info("Inicializando modelo...")
    try:
        model = NewModel()
    except Exception as e:
        logger.error(f"Erro ao inicializar modelo: {e}")
        sys.exit(1)
    
    # Valida cliente Label Studio
    if not model.client:
        logger.error("✗ Falha ao conectar com Label Studio. API Key inválida?")
        sys.exit(1)
    
    # Lista projetos
    logger.info("Buscando projetos do Label Studio...")
    projects = model.list_projects()
    
    if not projects:
        logger.error("Nenhum projeto encontrado no Label Studio")
        sys.exit(1)
    
    # Exibe projetos disponíveis
    print("\n" + "="*60)
    print("PROJETOS DISPONÍVEIS NO LABEL STUDIO")
    print("="*60)
    
    for idx, project in enumerate(projects, 1):
        print(f"{idx}. {project.title} (ID: {project.id})")
    
    print("="*60 + "\n")
    
    # Solicita seleção do usuário
    while True:
        try:
            choice = input("Selecione o número do projeto para fazer auto-labeling (ou 'q' para sair): ").strip()
            
            if choice.lower() == 'q':
                logger.info("Saindo...")
                sys.exit(0)
            
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(projects):
                selected_project = projects[choice_idx]
                break
            else:
                print("Opção inválida. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número ou 'q'.")
    
    # Confirma seleção
    print(f"\n✓ Projeto selecionado: {selected_project.title} (ID: {selected_project.id})")
    
    # Obtém tarefas não etiquetadas
    unlabeled_count = len(model.get_unlabeled_tasks(selected_project.id))
    
    if unlabeled_count == 0:
        logger.info(f"Nenhuma tarefa não etiquetada no projeto {selected_project.title}")
        sys.exit(0)
    
    print(f"Encontradas {unlabeled_count} imagens não etiquetadas.")
    
    # Confirma início do auto-labeling
    confirm = input(f"\nDeseja iniciar o auto-labeling de {unlabeled_count} imagens? (s/n): ").strip().lower()
    
    if confirm != 's':
        logger.info("Auto-labeling cancelado pelo usuário")
        sys.exit(0)
    
    print("\n" + "="*60)
    print("INICIANDO AUTO-LABELING...")
    print("="*60 + "\n")
    
    # Executa auto-labeling
    try:
        model.auto_label_project(selected_project.id)
        logger.info("✓ Auto-labeling concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante auto-labeling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
