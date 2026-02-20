import os
import logging
import json
import subprocess
import sys
import ctypes
import tempfile
import importlib.util
from uuid import uuid4
import requests
from io import BytesIO
from PIL import Image
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

logger = logging.getLogger(__name__)


def load_classes_config():
    candidates = [
        os.path.expanduser('../../../../classes.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'classes.json'),
        os.path.join(os.path.dirname(__file__), 'classes.json'),
        os.path.abspath(os.path.join(os.getcwd(), 'classes.json')),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            try:
                with open(p, 'r') as f:
                    cfg = json.load(f)
                logger.info(f"Carregado arquivo de classes: {p}")
                return cfg
            except Exception as e:
                logger.warning(f"Erro ao carregar {p}: {e}")
    logger.warning("Arquivo classes.json não encontrado. Usando fallback simples.")
    return {"classes": [], "filtering_rules": {"enabled": False}}


def load_image_from_url(url):
    token = os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN')
    headers = {"Authorization": f"Token {token}"} if token else {}
    
    # Adicionamos os headers para que o Label Studio permita o download da imagem
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert('RGB')

def get_absolute_url(url):
    if not url:
        return url
    # Se a URL já começar com http ou https, não faz nada
    if url.startswith('http://') or url.startswith('https://'):
        return url
    
    # Pega a URL base do ambiente
    base_url = os.getenv('LABEL_STUDIO_URL', 'http://127.0.0.1:8080').rstrip('/')
    
    # Garante que o caminho comece com /
    if not url.startswith('/'):
        url = '/' + url
        
    return f"{base_url}{url}"

class NewModel(LabelStudioMLBase):
    """Simple YOLOv11 backend compatible with the auto-labeling CLI.

    Detects objects with `ultralytics.YOLO` and maps detections to classes
    defined in `classes.json` using keyword matching and area heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Client(url=os.getenv('LABEL_STUDIO_URL'), api_key=os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN')) if (os.getenv('LABEL_STUDIO_URL') and (os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN'))) else None
        self.classes_config = load_classes_config()
        self._yolo = None
        self._vehicle_yolo = None
        self._plate_backend = None
        self._vehicle_backend = None
        self.device = 'cpu'
        self._use_half = False
        self.runtime_backend = 'cpu'
        self.ov_device = 'AUTO'
        self._plate_model_repair_attempted = False
        self._plate_model_load_failed = False
        # Resolve YOLO model path: prefer absolute/expanded env var, else package-local models/best.pt
        env_path = os.getenv('YOLO_MODEL_PATH')
        candidates = []
        if env_path:
            candidates.append(os.path.abspath(os.path.expanduser(env_path)))
        candidates.append(os.path.join(os.path.dirname(__file__), 'models', os.path.basename(env_path or 'best.pt')))
        # pick first existing candidate, else default to package-local absolute path
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = os.path.abspath(p)
                break
        self.model_path = found or os.path.abspath(candidates[-1])

    def _resolve_runtime_device(self):
        force_openvino = str(os.getenv('YOLO_FORCE_OPENVINO', '0')).strip().lower() in ('1', 'true', 'yes')
        # OpenVINO can change recall/precision compared to PT.
        # Enable explicitly with YOLO_ENABLE_OPENVINO=1 or YOLO_FORCE_OPENVINO=1.
        enable_openvino = str(os.getenv('YOLO_ENABLE_OPENVINO', '0')).strip().lower() in ('1', 'true', 'yes')
        try:
            if (not force_openvino) and torch is not None and torch.cuda.is_available():
                self.device = os.getenv('YOLO_DEVICE', 'cuda')
                self.runtime_backend = 'pt'
            elif enable_openvino:
                self.device = 'cpu'
                self.runtime_backend = 'openvino'
            else:
                self.device = os.getenv('YOLO_DEVICE_FALLBACK', 'cpu')
                self.runtime_backend = 'pt'
        except Exception:
            self.device = 'cpu'
            self.runtime_backend = 'pt'
        self.ov_device = self.device
        self._use_half = str(os.getenv('YOLO_USE_HALF', '1')).lower() in ('1', 'true', 'yes') and str(self.device).startswith('cuda')
        return self.device

    def _get_model_device(self, model):
        try:
            inner = getattr(model, 'model', None)
            if inner is None:
                return self.device
            if hasattr(inner, 'device'):
                return str(inner.device)
            if hasattr(inner, 'parameters'):
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
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullTotalPhys) / (1024.0 ** 3)
        except Exception:
            pass
        return 0.0

    def _is_low_memory_mode(self):
        forced = str(os.getenv('YOLO_LOW_MEM_MODE', '')).lower().strip()
        if forced in ('1', 'true', 'yes'):
            return True
        if forced in ('0', 'false', 'no'):
            return False
        ram_gb = self._get_total_ram_gb()
        if ram_gb > 0 and ram_gb <= float(os.getenv('YOLO_LOW_MEM_RAM_GB', 12)):
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
                for k in ('half', 'device', 'max_det', 'iou', 'imgsz'):
                    if k in args and ("'" + k + "'") in msg:
                        args.pop(k, None)
                        removed = True
                        break
                if removed:
                    continue
                raise
            except Exception as e:
                msg = str(e)
                if 'Invalid CUDA' in msg and 'device=' in msg:
                    args['device'] = 'cpu'
                    continue
                raise
        return model(source, verbose=False)

    def _export_openvino_if_needed(self, pt_model_path, model_tag):
        base_no_ext = os.path.splitext(pt_model_path)[0]
        default_dir = base_no_ext + '_openvino_model'
        explicit = os.getenv(f'YOLO_{model_tag.upper()}_OPENVINO_PATH')
        if explicit:
            explicit = os.path.abspath(os.path.expanduser(explicit))
            if os.path.exists(explicit):
                return explicit
        if os.path.exists(default_dir):
            return default_dir
        auto_export = str(os.getenv('YOLO_AUTO_EXPORT_OPENVINO', '1')).lower() in ('1', 'true', 'yes')
        if not auto_export:
            return None
        try:
            from ultralytics import YOLO
            logger.info('Exportando %s para OpenVINO: %s', model_tag, pt_model_path)
            exporter = YOLO(pt_model_path, task='detect')
            dynamic_export = str(os.getenv('YOLO_OPENVINO_DYNAMIC', '0')).lower() in ('1', 'true', 'yes')
            exported = exporter.export(format='openvino', dynamic=dynamic_export, half=False)
            if exported and os.path.exists(str(exported)):
                return str(exported)
        except Exception as e:
            logger.warning('Falha ao exportar %s para OpenVINO: %s', model_tag, e)
        if os.path.exists(default_dir):
            return default_dir
        return None

    def _repair_plate_model(self):
        script_path = os.path.join(os.path.dirname(__file__), 'download_model.py')
        if not os.path.exists(script_path):
            logger.error('Script de download do modelo de placas nao encontrado: %s', script_path)
            return False
        try:
            logger.warning('Tentando baixar novamente o modelo de placas: %s', self.model_path)
            subprocess.run([sys.executable, script_path], cwd=os.path.dirname(__file__), check=True)
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
                logger.info('Modelo de placas baixado novamente com sucesso.')
                return True
            logger.error('Download concluido, mas o arquivo de modelo nao foi encontrado: %s', self.model_path)
        except Exception as e:
            logger.error('Falha ao baixar novamente modelo de placas: %s', e)
        return False

    def setup(self):
        """Preload model (called by LabelStudioMLBase.__init__).
        To avoid long pauses on first prediction, the model is loaded here.
        You can set env YOLO_LOAD_IN_BACKGROUND=1 to load in background thread.
        """
        # determine device
        self._resolve_runtime_device()
        ram_gb = self._get_total_ram_gb()
        low_mem = self._is_low_memory_mode()
        logger.info(
            'Runtime profile: backend=%s device=%s ram_gb=%.2f low_mem=%s',
            self.runtime_backend,
            self.device,
            ram_gb,
            low_mem,
        )
        if not str(self.device).startswith('cuda'):
            print("CUDA não disponível, usando CPU")

        if os.getenv('YOLO_LOAD_IN_BACKGROUND', '0') == '1':
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
                logger.warning('Falha ao pré-carregar YOLO: %s', e)

    def _load_yolo(self):
        if self._yolo is not None and self._plate_backend == self.runtime_backend:
            return self._yolo
        if self._plate_model_load_failed:
            raise RuntimeError(f'Nao foi possivel carregar o modelo de placas: {self.model_path}')
        self._resolve_runtime_device()
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error('ultralytics not installed or import failed: %s', e)
            raise
        if not os.path.exists(self.model_path):
            logger.error('YOLO model file not found: %s', os.path.abspath(self.model_path))
            raise FileNotFoundError(self.model_path)
        # Guardrail: common failure is saving an HTML page (e.g. huggingface /blob URL) as .pt
        try:
            with open(self.model_path, 'rb') as f:
                head = f.read(16).lstrip()
            if head.startswith(b'<!doctype') or head.startswith(b'<html') or head.startswith(b'<?xml'):
                raise ValueError(
                    f"Arquivo de modelo invalido (HTML): {self.model_path}. "
                    "Baixe novamente com download_model.py (URL /resolve/, nao /blob/)."
                )
        except Exception:
            raise
        # load model and move to device if supported
        try:
            plate_path = self.model_path
            if self.runtime_backend == 'openvino':
                ov_path = self._export_openvino_if_needed(self.model_path, 'plate')
                if ov_path:
                    plate_path = ov_path
                else:
                    logger.warning('OpenVINO indisponivel para placa; usando .pt')
                    self.runtime_backend = 'pt'
            self._yolo = YOLO(plate_path, task='detect')
            # if ultralytics exposes .to(), try to move to device
            if self.runtime_backend == 'pt' and hasattr(self._yolo, 'to') and getattr(self, 'device', None):
                try:
                    self._yolo.to(self.device)
                except Exception:
                    pass
            self._plate_backend = self.runtime_backend
            logger.info(
                'YOLO model loaded from %s (device=%s, runtime=%s, half=%s, backend=%s)',
                os.path.abspath(plate_path),
                self._get_model_device(self._yolo),
                self.device,
                self._use_half,
                self.runtime_backend,
            )
        except Exception as e:
            err_msg = str(e)
            if 'PytorchStreamReader failed reading zip archive' in err_msg and not self._plate_model_repair_attempted:
                self._plate_model_repair_attempted = True
                logger.warning('Modelo de placas parece corrompido: %s', os.path.abspath(self.model_path))
                if self._repair_plate_model():
                    try:
                        self._yolo = YOLO(self.model_path, task='detect')
                        if self.runtime_backend == 'pt' and hasattr(self._yolo, 'to') and getattr(self, 'device', None):
                            try:
                                self._yolo.to(self.device)
                            except Exception:
                                pass
                        self._plate_backend = self.runtime_backend
                        logger.info(
                            'YOLO model loaded from %s (device=%s, runtime=%s, half=%s, backend=%s)',
                            os.path.abspath(self.model_path),
                            self._get_model_device(self._yolo),
                            self.device,
                            self._use_half,
                            self.runtime_backend,
                        )
                        return self._yolo
                    except Exception as retry_e:
                        logger.error('Erro ao carregar modelo YOLO apos novo download: %s', retry_e)
                self._plate_model_load_failed = True
                raise RuntimeError(
                    f'Modelo de placas invalido/corrompido em {os.path.abspath(self.model_path)}. '
                    'Baixe novamente com download_model.py.'
                ) from e
            self._plate_model_load_failed = True
            logger.error('Erro ao carregar modelo YOLO: %s', e)
            raise
        return self._yolo

    def _load_vehicle_yolo(self):
        if self._vehicle_yolo is not None and self._vehicle_backend == self.runtime_backend:
            return self._vehicle_yolo
        self._resolve_runtime_device()
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.error('ultralytics import failed for vehicle model: %s', e)
            raise
        # vehicle model path: allow override via env YOLO_VEHICLE_MODEL_PATH
        veh_env = os.getenv('YOLO_VEHICLE_MODEL_PATH')
        candidates = []
        if veh_env:
            candidates.append(os.path.abspath(os.path.expanduser(veh_env)))
        # fallback to examples/yolov11/models/yolo11x.pt
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov11', 'models', 'yolo11x.pt')))
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if not found:
            logger.error('Vehicle YOLO model not found. Checked: %s', candidates)
            raise FileNotFoundError('vehicle model not found')
        try:
            vehicle_path = found
            if self.runtime_backend == 'openvino':
                ov_path = self._export_openvino_if_needed(found, 'vehicle')
                if ov_path:
                    vehicle_path = ov_path
                else:
                    logger.warning('OpenVINO indisponivel para veiculo; usando .pt')
                    self.runtime_backend = 'pt'
            self._vehicle_yolo = YOLO(vehicle_path, task='detect')
            try:
                if self.runtime_backend == 'pt' and hasattr(self._vehicle_yolo, 'to'):
                    self._vehicle_yolo.to(self.device)
            except Exception:
                pass
            self._vehicle_backend = self.runtime_backend
            logger.info('Loaded vehicle YOLO model %s (device=%s, backend=%s)', vehicle_path, self._get_model_device(self._vehicle_yolo), self.runtime_backend)
        except Exception as e:
            logger.error('Failed to load vehicle YOLO model: %s', e)
            raise
        return self._vehicle_yolo

    def list_projects(self):
        if not self.client:
            logger.error('Cliente Label Studio não está disponível')
            return []
        try:
            return self.client.get_projects()
        except Exception as e:
            logger.error('Erro ao listar projetos: %s', e)
            return []

    def get_unlabeled_tasks(self, project_id):
        if not self.client:
            logger.error('Cliente Label Studio não está disponível')
            return []
        try:
            project = self.client.get_project(project_id)
            tasks = project.get_tasks()
            unlabeled = [t for t in tasks if not t.get('annotations') or t.get('is_labeled') == False]
            return unlabeled
        except Exception as e:
            logger.error('Erro ao buscar tarefas: %s', e)
            return []

    def _map_label_by_name(self, detected_name):
        detected_name = (detected_name or '').lower().strip()
        
        # mapping para garantir que placas nunca falhem
        plate_synonyms = ['placa', 'license', 'plate', 'car_plate', '0']
        if any(syn in detected_name for syn in plate_synonyms):
            return 'plate'

        # Tentativa via config JSON (para veículos: car, truck, bus...)
        classes = self.classes_config.get('classes', [])
        for cls in classes:
            if detected_name == cls.get('id', '').lower() or detected_name == cls.get('label', '').lower():
                return cls.get('label')
            for kw in cls.get('keywords', []):
                if kw.lower() in detected_name: 
                    return cls.get('label')    
        return None

    def _runtime_mode_label(self):
        self._resolve_runtime_device()
        if self.runtime_backend == 'openvino':
            return 'CPU+OpenVINO'
        if str(self.device).startswith('cuda'):
            has_trt = importlib.util.find_spec('tensorrt') is not None
            if has_trt:
                return 'CUDA+TensorRT'
            return 'CUDA+PyTorch (TensorRT indisponivel)'
        return 'CPU+PyTorch'
    
    def _predict_pil_image(self, img):
        w, h = img.size
        predictions_data = []

        # --- MODELO 1: PLACAS ---
        yolo_plate = self._load_yolo()
        conf_p = float(os.getenv('YOLO_CONF', 0.15))
        res_p_list = self._safe_model_predict(yolo_plate, img, conf=conf_p, verbose=False)

        for res_p in res_p_list:
            if hasattr(res_p, 'boxes') and res_p.boxes is not None:
                for det in res_p.boxes:
                    x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                    predictions_data.append({
                        'x': (x1 / w) * 100.0, 'y': (y1 / h) * 100.0,
                        'width': ((x2 - x1) / w) * 100.0, 'height': ((y2 - y1) / h) * 100.0,
                        'rotation': 0.0, 'score': float(det.conf.item()),
                        'label': 'plate', 'orig_w': w, 'orig_h': h
                    })

        # --- MODELO 2: VEÍCULOS ---
        try:
            v_yolo = self._load_vehicle_yolo()
            conf_v = float(os.getenv('YOLO_VEH_CONF', 0.25))
            global_vehicle_count = 0

            # 1. Detecção global
            res_v_list = self._safe_model_predict(v_yolo, img, conf=conf_v, verbose=False)
            for res_v in res_v_list:
                if res_v.boxes:
                    for det in res_v.boxes:
                        label = self._map_label_by_name(res_v.names.get(int(det.cls.item()), ""))
                        if label and not label.endswith('plate'):
                            x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                            predictions_data.append({
                                'x': (x1 / w) * 100.0, 'y': (y1 / h) * 100.0,
                                'width': ((x2 - x1) / w) * 100.0, 'height': ((y2 - y1) / h) * 100.0,
                                'rotation': 0.0, 'score': float(det.conf.item()),
                                'label': label, 'orig_w': w, 'orig_h': h
                            })
                            global_vehicle_count += 1

            # 2. Tiling condicional: por padrão, só executa se não houve veículo no passe global.
            tile_if_global_max = int(os.getenv('YOLO_TILE_IF_GLOBAL_MAX', 0))
            if global_vehicle_count <= tile_if_global_max and len(predictions_data) < int(os.getenv('YOLO_TILE_TRIGGER', 5)):
                tiles = [
                    (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                    (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)
                ]
                tile_conf = float(os.getenv('YOLO_TILE_CONF', conf_v + 0.1))
                for (tx1, ty1, tx2, ty2) in tiles:
                    tile_img = img.crop((tx1, ty1, tx2, ty2))
                    tile_res = self._safe_model_predict(v_yolo, tile_img, conf=tile_conf, verbose=False)
                    for r in tile_res:
                        if r.boxes:
                            for det in r.boxes:
                                label = self._map_label_by_name(r.names.get(int(det.cls.item()), ""))
                                if label and not label.endswith('plate'):
                                    bx1, by1, bx2, by2 = det.xyxy.cpu().numpy().flatten().tolist()
                                    predictions_data.append({
                                        'x': ((bx1 + tx1) / w) * 100.0, 'y': ((by1 + ty1) / h) * 100.0,
                                        'width': ((bx2 - bx1) / w) * 100.0, 'height': ((by2 - by1) / h) * 100.0,
                                        'rotation': 0.0, 'score': float(det.conf.item()),
                                        'label': label, 'orig_w': w, 'orig_h': h
                                    })
        except Exception as e:
            logger.error(f"Erro veículos: {e}")
            raise

        return {'items': predictions_data, 'width': w, 'height': h}

    def _predict_image(self, image_url):
        img = load_image_from_url(image_url)
        self._resolve_runtime_device()
        return self._predict_pil_image(img)

    def _predict_video(self, video_url):
        if cv2 is None:
            raise RuntimeError('opencv-python nao esta instalado; necessario para auto-labeling de video.')

        token = os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN')
        headers = {"Authorization": f"Token {token}"} if token else {}
        resp = requests.get(video_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()

        suffix = os.path.splitext(video_url.split('?')[0])[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_video_path = tf.name
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tf.write(chunk)

        frame_items = []
        frames_count = 0
        duration = 0.0
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise RuntimeError(f'Falha ao abrir video: {video_url}')

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if src_fps <= 0:
                src_fps = float(os.getenv('YOLO_VIDEO_SRC_FPS_FALLBACK', 30.0))
            frame_stride = int(os.getenv('YOLO_VIDEO_FRAME_STRIDE', 1))
            if frame_stride < 1:
                frame_stride = 1

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % frame_stride != 0:
                    continue

                h, w = frame.shape[:2]
                frames_count = max(frames_count, frame_idx)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pred = self._predict_pil_image(pil_img)
                time_sec = (frame_idx - 1) / src_fps
                duration = max(duration, time_sec)

                for it in pred['items']:
                    frame_items.append({
                        'label': it.get('label', 'object'),
                        'score': float(it.get('score', 0.0)),
                        'frame': int(frame_idx),
                        'time': float(time_sec),
                        'x': float(it.get('x', 0)),
                        'y': float(it.get('y', 0)),
                        'width': float(it.get('width', 0)),
                        'height': float(it.get('height', 0)),
                        'rotation': float(it.get('rotation', 0)),
                        'enabled': True,
                    })

            cap.release()
        finally:
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        return {
            'frame_items': frame_items,
            'frames_count': frames_count,
            'duration': duration
        }

    def _nms_boxes(self, boxes, scores, iou_threshold=0.3):
        """Simple NMS for xyxy boxes. boxes: Nx4 array, scores: N array."""
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes, dtype=float)
        scores = np.array(scores, dtype=float)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
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
        area = (box_arr[:, 2] - box_arr[:, 0] + 1.0) * (box_arr[:, 3] - box_arr[:, 1] + 1.0)
        nested_ioa = float(os.getenv('YOLO_VEH_NESTED_IOA', 0.80))
        score_margin = float(os.getenv('YOLO_VEH_NESTED_SCORE_MARGIN', 0.02))
        area_ratio_limit = float(os.getenv('YOLO_VEH_NESTED_AREA_RATIO_MAX', 0.65))
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
        raw = data.get('image') or data.get('image_url')
        media_type = 'image'
        if not raw and (data.get('video') or data.get('video_url')):
            raw = data.get('video') or data.get('video_url')
            media_type = 'video'
        if not raw and data:
            raw = list(data.values())[0]
        if isinstance(raw, str):
            lower = raw.lower().split('?', 1)[0]
            if lower.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                media_type = 'video'
        return media_type, get_absolute_url(raw)

    def _generate_video_results(self, video_pred):
        from_name = os.getenv('LS_VIDEO_FROM_NAME', 'label')
        to_name = os.getenv('LS_VIDEO_TO_NAME', 'video')
        results = []
        total_frames = max(1, int(video_pred.get('frames_count', 0)))
        duration = float(video_pred.get('duration', 0.0))
        for item in (video_pred.get('frame_items') or []):
            results.append({
                'id': str(uuid4()),
                'from_name': from_name,
                'to_name': to_name,
                'type': 'videorectangle',
                'score': float(item.get('score', 0.0)),
                'value': {
                    'labels': [item.get('label', 'object')],
                    'sequence': [{
                        'frame': int(item.get('frame', 1)),
                        'time': float(item.get('time', 0.0)),
                        'x': float(item.get('x', 0.0)),
                        'y': float(item.get('y', 0.0)),
                        'width': float(item.get('width', 0.0)),
                        'height': float(item.get('height', 0.0)),
                        'rotation': float(item.get('rotation', 0.0)),
                        'enabled': bool(item.get('enabled', True)),
                    }],
                    'framesCount': total_frames,
                    'duration': duration
                }
            })
        avg = float(np.mean([r.get('score', 0.0) for r in results])) if results else 0.0
        return [{'result': results, 'model_version': 'yolov11-video', 'score': avg}]

    
    def _generate_results(self, items):
        results = []
        total_score = 0.0

        for it in items:
            try:
                score = float(it.get('score', 0.0))
                label = it.get('label', '')
                res_item = {
                    'id': str(uuid4()),
                    'from_name': 'label',
                    'to_name': 'image',
                    'original_width': it.get('orig_w', 0),
                    'original_height': it.get('orig_h', 0),
                    'value': {
                        'x': it.get('x', 0),
                        'y': it.get('y', 0),
                        'width': it.get('width', 0),
                        'height': it.get('height', 0),
                        'rotation': it.get('rotation', 0),
                        'rectanglelabels': [label]
                    },
                    'score': score,
                    'type': 'rectanglelabels',
                    'readonly': False
                }
                results.append(res_item)
                total_score += score
            except Exception as e:
                logger.error(f"Erro ao formatar item de predição: {e}")
                continue

        avg = total_score / len(results) if results else 0
        # Retorna a estrutura correta para o Label Studio
        return [{'result': results, 'model_version': 'yolov11-combined', 'score': avg}]

    def _submit_predictions(self, task_id, predictions, project_id):
        if not self.client:
            logger.error('Cliente Label Studio não está disponível')
            return
        try:
            project = self.client.get_project(project_id)
            project.create_prediction(
                task_id=task_id,
                result=predictions[0]['result'],
                model_version=predictions[0].get('model_version', 'yolov11-auto-label'),
                score=predictions[0].get('score', 0)
            )
            logger.info('Predições submetidas para a tarefa %s', task_id)
        except Exception as e:
            logger.error('Erro ao submeter predições: %s', e)

    def auto_label_project(self, project_id):
        logger.info('Iniciando auto-labeling do projeto %s (YOLOv11)', project_id)
        logger.info('Runtime de inferencia: %s', self._runtime_mode_label())
        if not self.client:
            logger.error('Cliente Label Studio não está disponível')
            return
        
        project = self.client.get_project(project_id)
        tasks = self.get_unlabeled_tasks(project_id)
        success_count = 0
        error_count = 0
        
        for t in tasks:
            try:
                data = t.get('data') or {}
                media_type, media_url = self._resolve_task_media(data)
                if media_type == 'video':
                    pred = self._predict_video(media_url)
                    predictions = self._generate_video_results(pred)
                else:
                    pred = self._predict_image(media_url)
                    predictions = self._generate_results(pred['items'])
                
                self._submit_predictions(t.get('id'), predictions, project_id)
                success_count += 1
            except Exception as e:
                logger.error('Erro ao processar tarefa %s: %s', t.get('id'), e)
                error_count += 1

        logger.info(
            'Resumo auto-labeling projeto %s: sucesso=%s erro=%s total=%s',
            project_id,
            success_count,
            error_count,
            len(tasks),
        )
        if success_count == 0 and error_count > 0:
            raise RuntimeError('Nenhuma tarefa foi processada com sucesso.')

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            try:
                data = task.get('data') or {}
                media_type, media_url = self._resolve_task_media(data)
                if media_type == 'video':
                    pred_data = self._predict_video(media_url)
                    results = self._generate_video_results(pred_data)
                else:
                    pred_data = self._predict_image(media_url)
                    results = self._generate_results(pred_data['items'])
                
                if results:
                    predictions.append(results[0])
                else:
                    predictions.append({'result': []})
            except Exception as e:
                logger.error(f'Erro na predição: {e}')
                predictions.append({'result': []})
        return predictions
