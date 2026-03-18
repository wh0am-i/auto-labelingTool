import os
import sys
import requests

try:
    import torch
except Exception:
    torch = None

DEFAULT_MODEL_URL = (
    "https://huggingface.co/wh0am-i/yolov11x-BrPlate/resolve/main/best.pt"
)
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "best.pt")


def download_model(url=DEFAULT_MODEL_URL, out=DEFAULT_MODEL_PATH):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Baixando YOLO model de: {url}\n-> {out}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out, "wb") as f:
            dl = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
                    if total:
                        done = int(50 * dl / total)
                        bar = "[{}{}] {:.1f}%".format(
                            "=" * done,
                            " " * (50 - done),
                            dl / total * 100,
                        )
                        sys.stdout.write("\r" + bar + "   ")
                        sys.stdout.flush()
        if total:
            sys.stdout.write("\n")
        print("Download concluído.")
    except Exception as e:
        print(f"Erro ao baixar: {e}")


class Runtime:

    def _resolve_runtime_device(self):
        force_openvino = str(os.getenv("YOLO_FORCE_OPENVINO", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # OpenVINO can change recall/precision compared to PT.
        # Enable explicitly with YOLO_ENABLE_OPENVINO=1 or YOLO_FORCE_OPENVINO=1(force allways openvino).
        enable_openvino = str(
            os.getenv("YOLO_ENABLE_OPENVINO", "0")
        ).strip().lower() in ("1", "true", "yes")
        try:
            if (not force_openvino) and torch is not None and torch.cuda.is_available():
                self.device = os.getenv("YOLO_DEVICE", "cuda")
                self.runtime_backend = "pt"
            elif enable_openvino:
                self.device = "cpu"
                self.runtime_backend = "openvino"
            else:
                self.device = os.getenv("YOLO_DEVICE_FALLBACK", "cpu")
                self.runtime_backend = "pt"
        except Exception:
            self.device = "cpu"
            self.runtime_backend = "pt"
        self.ov_device = self.device
        self._use_half = str(os.getenv("YOLO_USE_HALF", "1")).lower() in (
            "1",
            "true",
            "yes",
        ) and str(self.device).startswith("cuda")
        return self.device

    def _sync_runtime(self):
        self._resolve_runtime_device()
        self.device = getattr(self, "device", self.device)
        self.runtime_backend = getattr(self, "runtime_backend", self.runtime_backend)
        self.ov_device = getattr(self, "ov_device", self.ov_device)
        self._use_half = getattr(self, "_use_half", self._use_half)


class Utils:

    @staticmethod
    def get_absolute_url(url):
        if not url:
            return url
        # Se a URL já começar com http ou https, não faz nada
        if url.startswith("http://") or url.startswith("https://"):
            return url

        # Pega a URL base do ambiente
        base_url = os.getenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080").rstrip("/")

        # Garante que o caminho comece com /
        if not url.startswith("/"):
            url = "/" + url

        return f"{base_url}{url}"

    def _resolve_task_media(self, data):
        data = data or {}
        raw = data.get("image") or data.get("image_url")
        media_type = "image"
        if not raw and (data.get("video") or data.get("video_url")):
            raw = data.get("video") or data.get("video_url")
            media_type = "video"
        if not raw and data:
            raw = list(data.values())[0]
        if isinstance(raw, str):
            lower = raw.lower().split("?", 1)[0]
            if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                media_type = "video"
        return media_type, self.get_absolute_url(raw)


def _print_usage():
    print(
        "Uso: python utils.py download_model [url] [out_path]\n"
        "Ex.: python utils.py download_model\n"
        "     python utils.py download_model https://.../best.pt ./models/best.pt"
    )


def _main():
    if len(sys.argv) < 2:
        _print_usage()
        return 1
    cmd = sys.argv[1].strip().lower().replace("-", "_")
    if cmd not in ("download_model", "download"):
        _print_usage()
        return 1
    url = sys.argv[2] if len(sys.argv) >= 3 else os.getenv("YOLO_MODEL_URL")
    out = sys.argv[3] if len(sys.argv) >= 4 else None
    download_model(url or DEFAULT_MODEL_URL, out or DEFAULT_MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
