import os
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from io import BytesIO
from PIL import Image
from uuid import uuid4
from runtime import Runtime

try:
    import cv2
except Exception:
    cv2 = None
try:
    import torch
except Exception:
    torch = None


class ImageProcessor:

    def __init__(self):
        self._yolo = None
        self.runtime = Runtime()

    def _predict_image(self, image_url):
        img = self.load_image_from_url(self, image_url)
        self.runtime._resolve_runtime_device()
        return self._predict_pil_image(img)

    def _predict_pil_image(self, img):
        w, h = img.size
        predictions_data = []

        # --- MODELO 1: PLACAS ---
        yolo_plate = self._load_yolo()
        conf_p = float(os.getenv("YOLO_CONF", 0.15))
        res_p_list = self._safe_model_predict(
            yolo_plate, img, conf=conf_p, verbose=False
        )

        for res_p in res_p_list:
            if hasattr(res_p, "boxes") and res_p.boxes is not None:
                for det in res_p.boxes:
                    x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                    predictions_data.append(
                        {
                            "x": (x1 / w) * 100.0,
                            "y": (y1 / h) * 100.0,
                            "width": ((x2 - x1) / w) * 100.0,
                            "height": ((y2 - y1) / h) * 100.0,
                            "rotation": 0.0,
                            "score": float(det.conf.item()),
                            "label": "plate",
                            "orig_w": w,
                            "orig_h": h,
                        }
                    )

        # --- MODELO 2: VEÍCULOS ---
        try:
            v_yolo = self._load_vehicle_yolo()
            conf_v = float(os.getenv("YOLO_VEH_CONF", 0.25))
            global_vehicle_count = 0

            # 1. Detecção global
            res_v_list = self._safe_model_predict(
                v_yolo, img, conf=conf_v, verbose=False
            )
            for res_v in res_v_list:
                if res_v.boxes:
                    for det in res_v.boxes:
                        label = self._map_label_by_name(
                            res_v.names.get(int(det.cls.item()), "")
                        )
                        if label and not label.endswith("plate"):
                            x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                            predictions_data.append(
                                {
                                    "x": (x1 / w) * 100.0,
                                    "y": (y1 / h) * 100.0,
                                    "width": ((x2 - x1) / w) * 100.0,
                                    "height": ((y2 - y1) / h) * 100.0,
                                    "rotation": 0.0,
                                    "score": float(det.conf.item()),
                                    "label": label,
                                    "orig_w": w,
                                    "orig_h": h,
                                }
                            )
                            global_vehicle_count += 1

            # 2. Tiling condicional: por padrão, só executa se não houve veículo no passe global.
            tile_if_global_max = int(os.getenv("YOLO_TILE_IF_GLOBAL_MAX", 0))
            if global_vehicle_count <= tile_if_global_max and len(
                predictions_data
            ) < int(os.getenv("YOLO_TILE_TRIGGER", 5)):
                tiles = [
                    (0, 0, w // 2, h // 2),
                    (w // 2, 0, w, h // 2),
                    (0, h // 2, w // 2, h),
                    (w // 2, h // 2, w, h),
                ]
                tile_conf = float(os.getenv("YOLO_TILE_CONF", conf_v + 0.1))
                for tx1, ty1, tx2, ty2 in tiles:
                    tile_img = img.crop((tx1, ty1, tx2, ty2))
                    tile_res = self._safe_model_predict(
                        v_yolo, tile_img, conf=tile_conf, verbose=False
                    )
                    for r in tile_res:
                        if r.boxes:
                            for det in r.boxes:
                                label = self._map_label_by_name(
                                    r.names.get(int(det.cls.item()), "")
                                )
                                if label and not label.endswith("plate"):
                                    bx1, by1, bx2, by2 = (
                                        det.xyxy.cpu().numpy().flatten().tolist()
                                    )
                                    predictions_data.append(
                                        {
                                            "x": ((bx1 + tx1) / w) * 100.0,
                                            "y": ((by1 + ty1) / h) * 100.0,
                                            "width": ((bx2 - bx1) / w) * 100.0,
                                            "height": ((by2 - by1) / h) * 100.0,
                                            "rotation": 0.0,
                                            "score": float(det.conf.item()),
                                            "label": label,
                                            "orig_w": w,
                                            "orig_h": h,
                                        }
                                    )
        except Exception as e:
            logger.error(f"Erro veículos: {e}")
            raise

        return {"items": predictions_data, "width": w, "height": h}

    def load_image_from_url(url):
        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        headers = {"Authorization": f"Token {token}"} if token else {}

        # Adicionamos os headers para que o Label Studio permita o download da imagem
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _generate_results(self, items):
        results = []
        total_score = 0.0

        for it in items:
            try:
                score = float(it.get("score", 0.0))
                label = it.get("label", "")
                res_item = {
                    "id": str(uuid4()),
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": it.get("orig_w", 0),
                    "original_height": it.get("orig_h", 0),
                    "value": {
                        "x": it.get("x", 0),
                        "y": it.get("y", 0),
                        "width": it.get("width", 0),
                        "height": it.get("height", 0),
                        "rotation": it.get("rotation", 0),
                        "rectanglelabels": [label],
                    },
                    "score": score,
                    "type": "rectanglelabels",
                    "readonly": False,
                }
                results.append(res_item)
                total_score += score
            except Exception as e:
                logger.error(f"Erro ao formatar item de predição: {e}")
                continue

        avg = total_score / len(results) if results else 0
        # Retorna a estrutura correta para o Label Studio
        return [{"result": results, "model_version": "yolov11-combined", "score": avg}]

    def _load_yolo(self):
        if self._yolo is not None and self._plate_backend == self.runtime_backend:
            return self._yolo
        if self._plate_model_load_failed:
            raise RuntimeError(
                f"Nao foi possivel carregar o modelo de placas: {self.model_path}"
            )
        self.runtime_backend._resolve_runtime_device()
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
        # Guardrail: common failure is saving an HTML page (e.g. huggingface /blob URL) as .pt
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
        # load model and move to device if supported
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
            # if ultralytics exposes .to(), try to move to device
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
                if self._repair_plate_model():
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
