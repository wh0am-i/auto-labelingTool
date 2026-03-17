import os

try:
    import torch
except Exception:
    torch = None


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
