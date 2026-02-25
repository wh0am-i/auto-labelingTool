#!/usr/bin/env python3

import os
import sys
import logging
import time
import argparse
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_label_studio_connection(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Conectado ao Label Studio com sucesso!")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Tentativa {attempt + 1}/{max_retries}: Aguardando Label Studio..."
                )
                time.sleep(2)
            continue
    logger.error(f"✗ Não foi possível conectar ao Label Studio em {url}")
    return False


def main():
    parser = argparse.ArgumentParser(description="YOLO auto-labeling CLI")
    parser.add_argument(
        "--model_path", help="Caminho para pesos do modelo de placa (YOLO_MODEL_PATH)"
    )
    parser.add_argument(
        "--vehicle_model_path",
        help="Caminho para pesos do modelo de veiculos (YOLO_VEHICLE_MODEL_PATH)",
    )
    args = parser.parse_args()

    if args.model_path:
        os.environ["YOLO_MODEL_PATH"] = os.path.abspath(
            os.path.expanduser(args.model_path)
        )
    if args.vehicle_model_path:
        os.environ["YOLO_VEHICLE_MODEL_PATH"] = os.path.abspath(
            os.path.expanduser(args.vehicle_model_path)
        )

    label_studio_url = os.getenv("LABEL_STUDIO_URL")
    if not label_studio_url:
        logger.error(
            "✗ LABEL_STUDIO_URL não configurado no ambiente. Defina LABEL_STUDIO_URL no .env"
        )
        sys.exit(1)
    api_key = os.getenv("LEGACY_TOKEN")
    if not api_key:
        logger.error("✗ LEGACY_TOKEN não configurado")
        logger.info(
            "Use: export LEGACY_TOKEN='seu_token'  (ou PERSONAL_TOKEN para a variante pessoal)"
        )
        sys.exit(1)

    logger.info(f"Verificando conexão com Label Studio em {label_studio_url}...")
    if not check_label_studio_connection(label_studio_url):
        logger.error("Impossível conectar ao Label Studio. Verifique se está rodando.")
        sys.exit(1)

    try:
        from model import NewModel
    except Exception as e:
        logger.error(f"Erro ao importar modelo: {e}")
        sys.exit(1)

    logger.info("Inicializando modelo YOLO...")
    try:
        model = NewModel()
    except Exception as e:
        logger.error(f"Erro ao inicializar modelo: {e}")
        sys.exit(1)

    if not model.client:
        logger.error("✗ Falha ao conectar com Label Studio. API Key inválida?")
        sys.exit(1)

    projects = model.list_projects()
    if not projects:
        logger.error("Nenhum projeto encontrado no Label Studio")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PROJETOS DISPONÍVEIS NO LABEL STUDIO")
    print("=" * 60)
    for idx, project in enumerate(projects, 1):
        print(f"{idx}. {project.title} (ID: {project.id})")
    print("=" * 60 + "\n")

    while True:
        try:
            choice = input(
                "Selecione o número do projeto para fazer auto-labeling (ou 'q' para sair): "
            ).strip()
            if choice.lower() == "q":
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

    print(
        f"\n✓ Projeto selecionado: {selected_project.title} (ID: {selected_project.id})"
    )
    unlabeled_count = len(model.get_unlabeled_tasks(selected_project.id))
    if unlabeled_count == 0:
        logger.info(
            f"Nenhuma tarefa não etiquetada no projeto {selected_project.title}"
        )
        sys.exit(0)
    print(f"Encontradas {unlabeled_count} tarefas não etiquetadas.")
    confirm = (
        input(f"\nDeseja iniciar o auto-labeling de {unlabeled_count} tarefas? (S/n): ")
        .strip()
        .lower()
    )
    if confirm == "n":
        logger.info("Auto-labeling cancelado pelo usuário")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("INICIANDO AUTO-LABELING...")
    print("=" * 60 + "\n")

    try:
        model.auto_label_project(selected_project.id)
        logger.info("✓ Auto-labeling concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante auto-labeling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
