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
from utils import Utils
from loader import Loader
from predict import Predictor

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


class NewModel(LabelStudioMLBase, Loader):

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
        self.runtime_backend = "pt"
        self.ov_device = "AUTO"
        self._plate_model_repair_attempted = False
        self._plate_model_load_failed = False
        Loader.__init__(self)
        self.runtime = Runtime()
        self.loader = self
        self.image_processor = ImageProcessor(self)
        self.video_processor = VideoProcessor(self.image_processor)
        self.utils = Utils()
        self.predictor = Predictor(
            model=self,
            image_processor=self.image_processor,
            video_processor=self.video_processor,
            utils=self.utils,
        )

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

    def _sync_runtime(self):
        self.runtime._resolve_runtime_device()
        self.device = getattr(self.runtime, "device", self.device)
        self.runtime_backend = getattr(
            self.runtime, "runtime_backend", self.runtime_backend
        )
        self.ov_device = getattr(self.runtime, "ov_device", self.ov_device)
        self._use_half = getattr(self.runtime, "_use_half", self._use_half)

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

    def setup(self):
        """Preload model (called by LabelStudioMLBase.__init__).
        To avoid long pauses on first prediction, the model is loaded here.
        You can set env YOLO_LOAD_IN_BACKGROUND=1 to load in background thread.
        """
        # determine device - Configure runtime based on device capability
        self._sync_runtime()
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
            t = threading.Thread(target=self.loader._load_yolo, daemon=True)
            t.start()
            # allow quick return while model loads in background
        else:
            try:
                self.loader._load_yolo()
                # warmup with a tiny image to avoid first-call slowdown
                if self._yolo is not None:
                    try:
                        dummy = np.zeros((1, 64, 64, 3), dtype=np.uint8)
                        self._yolo(dummy, verbose=False)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Falha ao pré-carregar YOLO: %s", e)

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
        self._sync_runtime()
        if self.runtime_backend == "openvino":
            return "CPU+OpenVINO"
        if str(self.device).startswith("cuda"):
            has_trt = importlib.util.find_spec("tensorrt") is not None
            if has_trt:
                return "CUDA+TensorRT"
            return "CUDA+PyTorch (TensorRT indisponivel)"
        return "CPU+PyTorch"

    def _submit_predictions(self, task_id, predictions, project_id):
        # Garante que há cliente conectado ao Label Studio
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return False
        try:
            # Recupera projeto para criar prediction via API legada
            project = self.client.get_project(project_id)
            # Cria prediction na task alvo
            project.create_prediction(
                task_id=task_id,
                result=predictions[0]["result"],
                model_version=predictions[0].get("model_version", "yolov11-auto-label"),
                score=predictions[0].get("score", 0),
            )
            logger.info("Predições submetidas para a tarefa %s", task_id)
            # Retorna sucesso para controle transacional do fluxo
            return True
        except Exception as e:
            logger.error("Erro ao submeter predições: %s", e)
            # Retorna falha para interromper remoção da task original
            return False

    def _import_converted_video_as_task(self, project_id, video_path):
        # Obtém projeto para importar arquivo convertido como nova task
        project = self.client.get_project(project_id)
        # Importa arquivo binário (upload) e recebe IDs das tasks criadas
        new_task_ids = project.import_tasks(video_path)
        if not new_task_ids:
            raise RuntimeError("Falha ao importar vídeo convertido como nova task.")
        # Seleciona a primeira task criada no upload
        new_task_id = int(new_task_ids[0])
        logger.info("Vídeo CFR importado em nova task: %s", new_task_id)
        return new_task_id

    def _cleanup_local_file(self, file_path):
        # Evita tentativa de remoção quando não há arquivo
        if not file_path:
            return
        try:
            # Remove arquivo temporário local
            os.remove(file_path)
        except Exception:
            # Ignora erro de limpeza para não interromper pipeline
            pass

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
            media_type = None
            converted_video_local_path = None
            try:
                source_task_id = int(t.get("id"))
                target_task_id = source_task_id

                # CENTRALIZAÇÃO: Chama o predictor para resolver a lógica de predição
                predictions, media_type, converted_video_local_path = (
                    self.predictor._get_predictions_for_task(t)
                )

                # Lógica específica de Vídeo CFR (Importação de nova task)
                if media_type == "video" and converted_video_local_path:
                    # Verifica se houve conversão real via metadados da predição (opcional, ou apenas se o path existir)
                    # Para manter compatibilidade total com seu código original:
                    target_task_id = self._import_converted_video_as_task(
                        project_id, converted_video_local_path
                    )

                # Submete predições na task alvo
                ok = self._submit_predictions(target_task_id, predictions, project_id)
                if not ok:
                    raise RuntimeError(
                        f"Falha ao submeter predição para task {target_task_id}"
                    )

                # Remoção da task original se necessário
                if (
                    media_type == "video"
                    and target_task_id != source_task_id
                    and str(os.getenv("YOLO_VIDEO_DELETE_ORIGINAL_TASK", "1")).lower()
                    in ("1", "true", "yes")
                ):
                    project.delete_task(source_task_id)
                    logger.info(
                        "Task original %s removida após conversão para CFR.",
                        source_task_id,
                    )

                success_count += 1
            except Exception as e:
                logger.error("Erro ao processar tarefa %s: %s", t.get("id"), e)
                error_count += 1
            finally:
                if media_type == "video":
                    self._cleanup_local_file(converted_video_local_path)

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
                media_type, media_url = self.utils._resolve_task_media(data)
                if media_type == "video":
                    # Predição de vídeo para endpoint síncrono /predict
                    pred_data = self.video_processor._predict_video(media_url)
                    results = self.video_processor._generate_video_results(pred_data)
                    # Limpeza do arquivo convertido local (não reaproveitado no /predict)
                    self._cleanup_local_file(pred_data.get("processed_video_path"))
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
