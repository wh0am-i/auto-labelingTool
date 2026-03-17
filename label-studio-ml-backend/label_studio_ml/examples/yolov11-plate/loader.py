import os
import logging

logger = logging.getLogger(__name__)


class Loader:

    def __init__(self, *args, **kwargs):

        env_path = os.getenv("YOLO_MODEL_PATH")
        candidates = []
        if env_path:
            candidates.append(os.path.abspath(os.path.expanduser(env_path)))
        candidates.append(
            os.path.join(
                os.path.dirname(__file__),
                "models",
                os.path.basename(env_path or "best.pt"),
            )
        )
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = os.path.abspath(p)
                break
        self.model_path = found or os.path.abspath(candidates[-1])

    def _load_yolo(self):
        if self._yolo is not None and self._plate_backend == self.runtime_backend:
            return self._yolo
        if self._plate_model_load_failed:
            raise RuntimeError(
                f"Nao foi possivel carregar o modelo de placas: {self.model_path}"
            )
        self._sync_runtime()
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error("ultralytics not installed or import failed: %s", e)
            raise
        if not os.path.exists(self.model_path):
            logger.error(
                "YOLO model file not found: %s", os.path.abspath(self.model_path)
            )
            raise FileNotFoundError(self.model_path)
        try:
            with open(self.model_path, "rb") as f:
                head = f.read(16).lstrip()
            if (
                head.startswith(b"<!doctype")
                or head.startswith(b"<html")
                or head.startswith(b"<?xml")
            ):
                raise ValueError(
                    f"Arquivo de modelo invalido (HTML): {self.model_path}. "
                    "Baixe novamente com download_model.py (URL /resolve/, nao /blob/)."
                )
        except Exception:
            raise
        try:
            plate_path = self.model_path
            if self.runtime_backend == "openvino":
                ov_path = self._export_openvino_if_needed(self.model_path, "plate")
                if ov_path:
                    plate_path = ov_path
                else:
                    logger.warning("OpenVINO indisponivel para placa; usando .pt")
                    self.runtime_backend = "pt"
            self._yolo = YOLO(plate_path, task="detect")
            if (
                self.runtime_backend == "pt"
                and hasattr(self._yolo, "to")
                and getattr(self, "device", None)
            ):
                try:
                    self._yolo.to(self.device)
                except Exception:
                    pass
            self._plate_backend = self.runtime_backend
            logger.info(
                "YOLO model loaded from %s (device=%s, runtime=%s, half=%s, backend=%s)",
                os.path.abspath(plate_path),
                self._get_model_device(self._yolo),
                self.device,
                self._use_half,
                self.runtime_backend,
            )
        except Exception as e:
            err_msg = str(e)
            if (
                "PytorchStreamReader failed reading zip archive" in err_msg
                and not self._plate_model_repair_attempted
            ):
                self._plate_model_repair_attempted = True
                logger.warning(
                    "Modelo de placas parece corrompido: %s",
                    os.path.abspath(self.model_path),
                )
                try:
                    self._yolo = YOLO(self.model_path, task="detect")
                    if (
                        self.runtime_backend == "pt"
                        and hasattr(self._yolo, "to")
                        and getattr(self, "device", None)
                    ):
                        try:
                            self._yolo.to(self.device)
                        except Exception:
                            pass
                    self._plate_backend = self.runtime_backend
                    logger.info(
                        "YOLO model loaded from %s (device=%s, runtime=%s, half=%s, backend=%s)",
                        os.path.abspath(self.model_path),
                        self._get_model_device(self._yolo),
                        self.device,
                        self._use_half,
                        self.runtime_backend,
                    )
                    return self._yolo
                except Exception as retry_e:
                    logger.error(
                        "Erro ao carregar modelo YOLO apos novo download: %s",
                        retry_e,
                    )
                self._plate_model_load_failed = True
                raise RuntimeError(
                    f"Modelo de placas invalido/corrompido em {os.path.abspath(self.model_path)}. "
                    "Baixe novamente com download_model.py."
                ) from e
            self._plate_model_load_failed = True
            logger.error("Erro ao carregar modelo YOLO: %s", e)
            raise
        return self._yolo

    def _load_vehicle_yolo(self):
        if (
            self._vehicle_yolo is not None
            and self._vehicle_backend == self.runtime_backend
        ):
            return self._vehicle_yolo
        self._sync_runtime()
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error("ultralytics import failed for vehicle model: %s", e)
            raise
        # vehicle model path: allow override via env YOLO_VEHICLE_MODEL_PATH
        veh_env = os.getenv("YOLO_VEHICLE_MODEL_PATH")
        candidates = []
        if veh_env:
            candidates.append(os.path.abspath(os.path.expanduser(veh_env)))
        # fallback to examples/yolov11/models/yolo11x.pt
        candidates.append(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "yolov11", "models", "yolo11x.pt"
                )
            )
        )
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if not found:
            logger.error("Vehicle YOLO model not found. Checked: %s", candidates)
            raise FileNotFoundError("vehicle model not found")
        try:
            vehicle_path = found
            if self.runtime_backend == "openvino":
                ov_path = self._export_openvino_if_needed(found, "vehicle")
                if ov_path:
                    vehicle_path = ov_path
                else:
                    logger.warning("OpenVINO indisponivel para veiculo; usando .pt")
                    self.runtime_backend = "pt"
            self._vehicle_yolo = YOLO(vehicle_path, task="detect")
            try:
                if self.runtime_backend == "pt" and hasattr(self._vehicle_yolo, "to"):
                    self._vehicle_yolo.to(self.device)
            except Exception:
                pass
            self._vehicle_backend = self.runtime_backend
            logger.info(
                "Loaded vehicle YOLO model %s (device=%s, backend=%s)",
                vehicle_path,
                self._get_model_device(self._vehicle_yolo),
                self.runtime_backend,
            )
        except Exception as e:
            logger.error("Failed to load vehicle YOLO model: %s", e)
            raise
        return self._vehicle_yolo
