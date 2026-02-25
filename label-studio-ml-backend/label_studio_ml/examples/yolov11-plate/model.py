import os
import logging
import json
import subprocess
import sys
import ctypes
import importlib.util
from uuid import uuid4
import numpy as np
import threading

try:
    import cv2
except Exception:
    cv2 = None
try:
    import torch
except Exception:
    torch = None
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import Client
from video import VideoProcessor
from image import ImageProcessor
from runtime import Runtime


logger = logging.getLogger(__name__)


def load_classes_config():
    candidates = [
        os.path.expanduser("../../../../classes.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "classes.json"),
        os.path.join(os.path.dirname(__file__), "classes.json"),
        os.path.abspath(os.path.join(os.getcwd(), "classes.json")),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    cfg = json.load(f)
                logger.info(f"Carregado arquivo de classes: {p}")
                return cfg
            except Exception as e:
                logger.warning(f"Erro ao carregar {p}: {e}")
    logger.warning("Arquivo classes.json não encontrado. Usando fallback simples.")
    return {"classes": [], "filtering_rules": {"enabled": False}}


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


class NewModel(LabelStudioMLBase):
    """Simple YOLOv11 backend compatible with the auto-labeling CLI.

    Detects objects with `ultralytics.YOLO` and maps detections to classes
    defined in `classes.json` using keyword matching and area heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = (
            Client(
                url=os.getenv("LABEL_STUDIO_URL"),
                api_key=os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN"),
            )
            if (
                os.getenv("LABEL_STUDIO_URL")
                and (os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN"))
            )
            else None
        )
        self.classes_config = load_classes_config()
        self._yolo = None
        self._vehicle_yolo = None
        self._plate_backend = None
        self._vehicle_backend = None
        self.device = "cpu"
        self._use_half = False
        self.runtime_backend = "cpu"
        self.ov_device = "AUTO"
        self._plate_model_repair_attempted = False
        self._plate_model_load_failed = False
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        self.runtime_backend = Runtime()
        # Resolve YOLO model path: prefer absolute/expanded env var, else package-local models/best.pt
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
        # pick first existing candidate, else default to package-local absolute path
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = os.path.abspath(p)
                break
        self.model_path = found or os.path.abspath(candidates[-1])

    def _get_model_device(self, model):
        try:
            inner = getattr(model, "model", None)
            if inner is None:
                return self.device
            if hasattr(inner, "device"):
                return str(inner.device)
            if hasattr(inner, "parameters"):
                params = list(inner.parameters())
                if params:
                    return str(params[0].device)
        except Exception:
            pass
        return self.device

    def _get_total_ram_gb(self):
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullTotalPhys) / (1024.0**3)
        except Exception:
            pass
        return 0.0

    def _is_low_memory_mode(self):
        forced = str(os.getenv("YOLO_LOW_MEM_MODE", "")).lower().strip()
        if forced in ("1", "true", "yes"):
            return True
        if forced in ("0", "false", "no"):
            return False
        ram_gb = self._get_total_ram_gb()
        if ram_gb > 0 and ram_gb <= float(os.getenv("YOLO_LOW_MEM_RAM_GB", 12)):
            return True
        return False

    def _safe_model_predict(self, model, source, **kwargs):
        args = dict(kwargs)
        for _ in range(8):
            try:
                return model(source, **args)
            except TypeError as e:
                msg = str(e)
                removed = False
                for k in ("half", "device", "max_det", "iou", "imgsz"):
                    if k in args and ("'" + k + "'") in msg:
                        args.pop(k, None)
                        removed = True
                        break
                if removed:
                    continue
                raise
            except Exception as e:
                msg = str(e)
                if "Invalid CUDA" in msg and "device=" in msg:
                    args["device"] = "cpu"
                    continue
                raise
        return model(source, verbose=False)

    def _export_openvino_if_needed(self, pt_model_path, model_tag):
        base_no_ext = os.path.splitext(pt_model_path)[0]
        default_dir = base_no_ext + "_openvino_model"
        explicit = os.getenv(f"YOLO_{model_tag.upper()}_OPENVINO_PATH")
        if explicit:
            explicit = os.path.abspath(os.path.expanduser(explicit))
            if os.path.exists(explicit):
                return explicit
        if os.path.exists(default_dir):
            return default_dir
        auto_export = str(os.getenv("YOLO_AUTO_EXPORT_OPENVINO", "1")).lower() in (
            "1",
            "true",
            "yes",
        )
        if not auto_export:
            return None
        try:
            from ultralytics import YOLO

            logger.info("Exportando %s para OpenVINO: %s", model_tag, pt_model_path)
            exporter = YOLO(pt_model_path, task="detect")
            dynamic_export = str(os.getenv("YOLO_OPENVINO_DYNAMIC", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
            exported = exporter.export(
                format="openvino", dynamic=dynamic_export, half=False
            )
            if exported and os.path.exists(str(exported)):
                return str(exported)
        except Exception as e:
            logger.warning("Falha ao exportar %s para OpenVINO: %s", model_tag, e)
        if os.path.exists(default_dir):
            return default_dir
        return None

    def _repair_plate_model(self):
        script_path = os.path.join(os.path.dirname(__file__), "download_model.py")
        if not os.path.exists(script_path):
            logger.error(
                "Script de download do modelo de placas nao encontrado: %s", script_path
            )
            return False
        try:
            logger.warning(
                "Tentando baixar novamente o modelo de placas: %s", self.model_path
            )
            subprocess.run(
                [sys.executable, script_path], cwd=os.path.dirname(__file__), check=True
            )
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
                logger.info("Modelo de placas baixado novamente com sucesso.")
                return True
            logger.error(
                "Download concluido, mas o arquivo de modelo nao foi encontrado: %s",
                self.model_path,
            )
        except Exception as e:
            logger.error("Falha ao baixar novamente modelo de placas: %s", e)
        return False

    def setup(self):
        """Preload model (called by LabelStudioMLBase.__init__).
        To avoid long pauses on first prediction, the model is loaded here.
        You can set env YOLO_LOAD_IN_BACKGROUND=1 to load in background thread.
        """
        # determine device
        self.runtime_backend._resolve_runtime_device()
        ram_gb = self._get_total_ram_gb()
        low_mem = self._is_low_memory_mode()
        logger.info(
            "Runtime profile: backend=%s device=%s ram_gb=%.2f low_mem=%s",
            self.runtime_backend,
            self.device,
            ram_gb,
            low_mem,
        )
        if not str(self.device).startswith("cuda"):
            print("CUDA não disponível, usando CPU")

        if os.getenv("YOLO_LOAD_IN_BACKGROUND", "0") == "1":
            t = threading.Thread(target=self._load_yolo, daemon=True)
            t.start()
            # allow quick return while model loads in background
        else:
            try:
                self._load_yolo()
                # warmup with a tiny image to avoid first-call slowdown
                if self._yolo is not None:
                    try:
                        dummy = np.zeros((1, 64, 64, 3), dtype=np.uint8)
                        self._yolo(dummy, verbose=False)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Falha ao pré-carregar YOLO: %s", e)

    def _load_vehicle_yolo(self):
        if (
            self._vehicle_yolo is not None
            and self._vehicle_backend == self.runtime_backend
        ):
            return self._vehicle_yolo
        self.runtime_backend._resolve_runtime_device()
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

    def list_projects(self):
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return []
        try:
            return self.client.get_projects()
        except Exception as e:
            logger.error("Erro ao listar projetos: %s", e)
            return []

    def get_unlabeled_tasks(self, project_id):
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return []
        try:
            project = self.client.get_project(project_id)
            tasks = project.get_tasks()
            unlabeled = [
                t
                for t in tasks
                if not t.get("annotations") or t.get("is_labeled") == False
            ]
            return unlabeled
        except Exception as e:
            logger.error("Erro ao buscar tarefas: %s", e)
            return []

    def _map_label_by_name(self, detected_name):
        detected_name = (detected_name or "").lower().strip()

        # mapping para garantir que placas nunca falhem
        plate_synonyms = ["placa", "license", "plate", "car_plate", "0"]
        if any(syn in detected_name for syn in plate_synonyms):
            return "plate"

        # Tentativa via config JSON (para veículos: car, truck, bus...)
        classes = self.classes_config.get("classes", [])
        for cls in classes:
            if (
                detected_name == cls.get("id", "").lower()
                or detected_name == cls.get("label", "").lower()
            ):
                return cls.get("label")
            for kw in cls.get("keywords", []):
                if kw.lower() in detected_name:
                    return cls.get("label")
        return None

    def _runtime_mode_label(self):
        self.runtime_backend._resolve_runtime_device()
        if self.runtime_backend == "openvino":
            return "CPU+OpenVINO"
        if str(self.device).startswith("cuda"):
            has_trt = importlib.util.find_spec("tensorrt") is not None
            if has_trt:
                return "CUDA+TensorRT"
            return "CUDA+PyTorch (TensorRT indisponivel)"
        return "CPU+PyTorch"

    def _nms_boxes(self, boxes, scores, iou_threshold=0.3):
        """Simple NMS for xyxy boxes. boxes: Nx4 array, scores: N array."""
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes, dtype=float)
        scores = np.array(scores, dtype=float)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def _suppress_nested_vehicle_boxes(self, boxes, scores, labels):
        """Remove lower-score nested vehicle boxes from same class."""
        if not boxes:
            return []
        box_arr = np.array(boxes, dtype=float)
        area = (box_arr[:, 2] - box_arr[:, 0] + 1.0) * (
            box_arr[:, 3] - box_arr[:, 1] + 1.0
        )
        nested_ioa = float(os.getenv("YOLO_VEH_NESTED_IOA", 0.80))
        score_margin = float(os.getenv("YOLO_VEH_NESTED_SCORE_MARGIN", 0.02))
        area_ratio_limit = float(os.getenv("YOLO_VEH_NESTED_AREA_RATIO_MAX", 0.65))
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            for j in range(len(boxes)):
                if i == j or not keep[j]:
                    continue
                if str(labels[i]).lower() != str(labels[j]).lower():
                    continue
                xi1, yi1, xi2, yi2 = box_arr[i]
                xj1, yj1, xj2, yj2 = box_arr[j]
                # Intersection box i ∩ j
                xx1 = max(xi1, xj1)
                yy1 = max(yi1, yj1)
                xx2 = min(xi2, xj2)
                yy2 = min(yi2, yj2)
                iw = max(0.0, xx2 - xx1 + 1.0)
                ih = max(0.0, yy2 - yy1 + 1.0)
                inter = iw * ih
                if inter <= 0:
                    continue
                # ioa of smaller over itself
                if area[i] <= area[j]:
                    small_idx, big_idx = i, j
                else:
                    small_idx, big_idx = j, i
                ioa = inter / max(1.0, area[small_idx])
                if ioa < nested_ioa:
                    continue
                if (area[small_idx] / max(1.0, area[big_idx])) > area_ratio_limit:
                    continue
                # Smaller center must be inside bigger.
                sx1, sy1, sx2, sy2 = box_arr[small_idx]
                bx1, by1, bx2, by2 = box_arr[big_idx]
                scx = 0.5 * (sx1 + sx2)
                scy = 0.5 * (sy1 + sy2)
                if not (bx1 <= scx <= bx2 and by1 <= scy <= by2):
                    continue
                # Suppress smaller if it is not clearly higher score.
                if scores[small_idx] <= (scores[big_idx] + score_margin):
                    keep[small_idx] = False
        return [idx for idx, ok in enumerate(keep) if ok]

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
        return media_type, get_absolute_url(raw)

    def _submit_predictions(self, task_id, predictions, project_id):
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return
        try:
            project = self.client.get_project(project_id)
            project.create_prediction(
                task_id=task_id,
                result=predictions[0]["result"],
                model_version=predictions[0].get("model_version", "yolov11-auto-label"),
                score=predictions[0].get("score", 0),
            )
            logger.info("Predições submetidas para a tarefa %s", task_id)
        except Exception as e:
            logger.error("Erro ao submeter predições: %s", e)

    def auto_label_project(self, project_id):
        logger.info("Iniciando auto-labeling do projeto %s (YOLOv11)", project_id)
        logger.info("Runtime de inferencia: %s", self._runtime_mode_label())
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return

        project = self.client.get_project(project_id)
        tasks = self.get_unlabeled_tasks(project_id)
        success_count = 0
        error_count = 0

        for t in tasks:
            try:
                data = t.get("data") or {}
                media_type, media_url = self._resolve_task_media(data)
                if media_type == "video":
                    pred = self.video_processor._predict_video(media_url)
                    predictions = self.video_processor._generate_video_results(pred)
                else:
                    pred = self.image_processor._predict_image(media_url)
                    predictions = self.image_processor._generate_results(pred["items"])

                self._submit_predictions(t.get("id"), predictions, project_id)
                success_count += 1
            except Exception as e:
                logger.error("Erro ao processar tarefa %s: %s", t.get("id"), e)
                error_count += 1

        logger.info(
            "Resumo auto-labeling projeto %s: sucesso=%s erro=%s total=%s",
            project_id,
            success_count,
            error_count,
            len(tasks),
        )
        if success_count == 0 and error_count > 0:
            raise RuntimeError("Nenhuma tarefa foi processada com sucesso.")

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            try:
                data = task.get("data") or {}
                media_type, media_url = self._resolve_task_media(data)
                if media_type == "video":
                    pred_data = self.video_processor._predict_video(media_url)
                    results = self.video_processor._generate_video_results(pred_data)
                else:
                    pred_data = self.image_processor._predict_image(media_url)
                    results = self.image_processor._generate_results(pred_data["items"])

                if results:
                    predictions.append(results[0])
                else:
                    predictions.append({"result": []})
            except Exception as e:
                logger.error(f"Erro na predição: {e}")
                predictions.append({"result": []})
        return predictions
