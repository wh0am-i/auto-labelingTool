#!/usr/bin/env python3
"""
Script para baixar checkpoint SAM2 (antigo) compatível com sam2_hiera_l config
"""

import os
import urllib.request
import sys

# SAM2 checkpoint URLs 
CHECKPOINTS = {
    'sam2.1_hiera_large.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
    'sam2.1_hiera_base_plus.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
    'sam2.1_hiera_small.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
    'sam2.1_hiera_tiny.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
}

OUTPUT_DIR = os.path.expanduser('./label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/checkpoints')

def download_checkpoint(name, url):
    """Baixa checkpoint com barra de progresso"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, name)
    
    if os.path.exists(filepath):
        print(f"✓ {name} já existe em {filepath}")
        return True
    
    print(f"\n📥 Baixando {name}...")
    print(f"   URL: {url}")
    print(f"   Destino: {filepath}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 // total_size, 100)
            bar_length = 40
            filled = int(bar_length * percent // 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            sys.stdout.write(f'\r[{bar}] {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✓ {name} baixado com sucesso!")
        return True
    except Exception as e:
        print(f"\n✗ Erro ao baixar {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("SAM2 Checkpoint Downloader")
    print("=" * 60)
    print(f"\nDestino: {OUTPUT_DIR}")
    
    print("\nCheckpoints disponíveis:")
    for idx, name in enumerate(CHECKPOINTS.keys(), 1):
        print(f"  {idx}. {name}")
    
    while True:
        try:
            choice = input("\nEscolha (1-4) ou 'a' para todos, 'q' para sair: ").strip().lower()
            
            if choice == 'q':
                print("Cancelado.")
                return
            elif choice == 'a':
                for name, url in CHECKPOINTS.items():
                    download_checkpoint(name, url)
                break
            elif choice in ['1', '2', '3']:
                names = list(CHECKPOINTS.keys())
                name = names[int(choice) - 1]
                url = CHECKPOINTS[name]
                download_checkpoint(name, url)
                break
            else:
                print("Opção inválida")
        except KeyboardInterrupt:
            print("\nCancelado.")
            return

if __name__ == '__main__':
    main()
