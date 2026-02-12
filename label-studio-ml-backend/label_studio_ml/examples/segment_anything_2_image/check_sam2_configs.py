#!/usr/bin/env python3
"""
Script para descobrir os nomes de configs disponíveis no SAM2
Execute: python check_sam2_configs.py
"""

import os
import sys

try:
    import sam2
    sam2_path = os.path.dirname(sam2.__file__)
    configs_path = os.path.join(sam2_path, 'configs')
    
    print(f"SAM2 instalado em: {sam2_path}")
    print(f"Pasta de configs: {configs_path}")
    print("\nConfigs disponíveis:")
    
    if os.path.exists(configs_path):
        for root, dirs, files in os.walk(configs_path):
            for f in files:
                if f.endswith('.yaml'):
                    rel_path = os.path.relpath(os.path.join(root, f), configs_path)
                    config_name = rel_path.replace('.yaml', '').replace(os.sep, '/')
                    print(f"  ✓ {config_name}")
    else:
        print(f"  ✗ Pasta de configs não encontrada em {configs_path}")
        
except ImportError:
    print("SAM2 não está instalado, está com o venv ativado?. Execute: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)
