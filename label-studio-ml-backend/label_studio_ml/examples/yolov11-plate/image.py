import os
import logging
from uuid import uuid4
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _plate_center_inside_any_vehicle(plate_xyxy, vehicle_boxes):
        x1, y1, x2, y2 = plate_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                return True
        return False

    def _predict_pil_image(self, img):
        w, h = img.size

        # Detecções
        plate_candidates = self._detect_plates(img)
        vehicle_results = self._detect_vehicles_with_tiling(img)

        # Filtro de proximidade
        filtered_plates = self._filter_plates(
            plate_candidates, vehicle_results["boxes"], w, h
        )

        all_items = vehicle_results["predictions"] + filtered_plates

        # LIMPEZA FINAL: Aplica NMS em tudo para remover sobreposições do Tiling
        cleaned_items = self._apply_nms(all_items)

        return {
            "items": cleaned_items,
            "width": w,
            "height": h,
        }

    def _detect_plates(self, img):
        yolo_plate = self.model._load_yolo()
        conf_p = float(os.getenv("YOLO_CONF", 0.08))
        res_list = self.model._safe_model_predict(
            yolo_plate, img, conf=conf_p, verbose=False
        )

        candidates = []
        for res in res_list:
            if hasattr(res, "boxes"):
                for det in res.boxes:
                    candidates.append(
                        (
                            det.xyxy.cpu().numpy().flatten().tolist(),
                            float(det.conf.item()),
                        )
                    )
        return candidates

    def _detect_vehicles_with_tiling(self, img):
        w, h = img.size
        v_yolo = self.model._load_vehicle_yolo()
        conf_v = float(os.getenv("YOLO_VEH_CONF", 0.10))

        predictions_data = []
        vehicle_boxes = []
        global_vehicle_count = 0

        # Detecção Normal
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
                        vehicle_boxes.append((x1, y1, x2, y2))
                        predictions_data.append(
                            self._format_item(
                                x1, y1, x2, y2, w, h, label, float(det.conf.item())
                            )
                        )
                        global_vehicle_count += 1

        # Estratégia de Tiling
        tile_if_global_max = int(os.getenv("YOLO_TILE_IF_GLOBAL_MAX", 30))
        tile_trigger = int(os.getenv("YOLO_TILE_TRIGGER", 10))

        if (
            global_vehicle_count <= tile_if_global_max
            and len(predictions_data) < tile_trigger
        ):
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
                                real_x1, real_y1 = bx1 + tx1, by1 + ty1
                                real_x2, real_y2 = bx2 + tx1, by2 + ty1
                                vehicle_boxes.append(
                                    (real_x1, real_y1, real_x2, real_y2)
                                )
                                predictions_data.append(
                                    self._format_item(
                                        real_x1,
                                        real_y1,
                                        real_x2,
                                        real_y2,
                                        w,
                                        h,
                                        label,
                                        float(det.conf.item()),
                                    )
                                )

        return {"predictions": predictions_data, "boxes": vehicle_boxes}

    def _format_item(self, x1, y1, x2, y2, w, h, label, score):
        return {
            "x": (x1 / w) * 100.0,
            "y": (y1 / h) * 100.0,
            "width": ((x2 - x1) / w) * 100.0,
            "height": ((y2 - y1) / h) * 100.0,
            "rotation": 0.0,
            "score": score,
            "label": label,
            "orig_w": w,
            "orig_h": h,
        }

    def _apply_nms(self, items, iou_threshold=0.25):
        """Unificado: Remove detecções sobrepostas baseadas em IoU."""
        if not items:
            return []

        # Ordena por score descendente
        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        keep = []

        while sorted_items:
            best = sorted_items.pop(0)
            keep.append(best)

            remaining = []
            for item in sorted_items:
                # Se forem da mesma classe e sobrepuserem muito, descarta o pior
                if (
                    item["label"] == best["label"]
                    and self._calculate_iou(best, item) > iou_threshold
                ):
                    continue
                remaining.append(item)
            sorted_items = remaining

        return keep

    def _calculate_iou(self, box1, box2):
        """Calcula IoU usando as porcentagens do Label Studio."""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _filter_plates(self, candidates, vehicle_boxes, w, h):
        final_plates = []
        conf_inside = float(os.getenv("YOLO_PLATE_CONF_IN_VEHICLE", 0.08))
        conf_outside = float(os.getenv("YOLO_PLATE_CONF_OUTSIDE_VEHICLE", 0.10))

        for box, score in candidates:
            is_inside = self._plate_center_inside_any_vehicle(box, vehicle_boxes)
            threshold = conf_inside if is_inside else conf_outside

            if score >= threshold:
                x1, y1, x2, y2 = box
                final_plates.append(
                    self._format_item(x1, y1, x2, y2, w, h, "plate", score)
                )
        return final_plates

    def _generate_results(self, items):
        results = []
        total_score = 0.0
        for it in items:
            try:
                score = float(it.get("score", 0.0))
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
                        "rectanglelabels": [it.get("label", "")],
                    },
                    "score": score,
                    "type": "rectanglelabels",
                    "readonly": False,
                }
                results.append(res_item)
                total_score += score
            except Exception:
                continue
        avg = total_score / len(results) if results else 0
        return [{"result": results, "model_version": "yolov11x", "score": avg}]
