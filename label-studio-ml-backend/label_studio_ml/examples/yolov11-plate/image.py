import os
import logging
from io import BytesIO
from uuid import uuid4

import requests
from PIL import Image
from loader import Loader
from model import NewModel

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.loader = Loader
        self.model = NewModel()

    def _predict_image(self, image_url):
        img = self.load_image_from_url(image_url)
        return self._predict_pil_image(img)

    def _predict_pil_image(self, img):
        w, h = img.size
        predictions_data = []

        yolo_plate = self.model._load_yolo()
        conf_p = float(os.getenv("YOLO_CONF", 0.15))
        res_p_list = self.model._safe_model_predict(
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

        try:
            v_yolo = self.loader._load_vehicle_yolo()
            conf_v = float(os.getenv("YOLO_VEH_CONF", 0.25))
            global_vehicle_count = 0
            res_v_list = self.model._safe_model_predict(
                v_yolo, img, conf=conf_v, verbose=False
            )
            for res_v in res_v_list:
                if res_v.boxes:
                    for det in res_v.boxes:
                        label = self.model._map_label_by_name(
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
                    tile_res = self.model._safe_model_predict(
                        v_yolo, tile_img, conf=tile_conf, verbose=False
                    )
                    for r in tile_res:
                        if r.boxes:
                            for det in r.boxes:
                                label = self.model._map_label_by_name(
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
            logger.error("Erro veiculos: %s", e)
            raise

        return {"items": predictions_data, "width": w, "height": h}

    @staticmethod
    def load_image_from_url(url):
        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        headers = {"Authorization": f"Token {token}"} if token else {}
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
                logger.error("Erro ao formatar item de predição: %s", e)
                continue

        avg = total_score / len(results) if results else 0
        return [{"result": results, "model_version": "yolov11-combined", "score": avg}]
