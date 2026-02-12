#!/usr/bin/env python3
import os
import requests
import sys

DEFAULT_URL = 'https://huggingface.co/wh0am-i/yolov11x-BrPlate/tree/main/best.pt'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'models')
OUT_PATH = os.path.join(OUT_DIR, 'best.pt')


def download(url=DEFAULT_URL, out=OUT_PATH):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Baixando YOLO model de: {url}\n-> {out}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(out, 'wb') as f:
            dl = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
                    if total:
                        done = int(50 * dl / total)
                        sys.stdout.write('\r[{}{}] {:.1f}%'.format('=' * done, ' ' * (50 - done), dl / total * 100))
                        sys.stdout.flush()
        print('\nDownload concluído.')
    except Exception as e:
        print(f'Erro ao baixar: {e}')


if __name__ == '__main__':
    url = os.getenv('YOLO_MODEL_URL') or DEFAULT_URL
    download(url)
