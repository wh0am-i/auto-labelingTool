import os
import logging
import json
from uuid import uuid4
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import threading
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

    def setup(self):
        """Preload model (called by LabelStudioMLBase.__init__).
        To avoid long pauses on first prediction, the model is loaded here.
        You can set env YOLO_LOAD_IN_BACKGROUND=1 to load in background thread.
        """
        # determine device
        try:
            self.device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
        except Exception:
            self.device = 'cpu'

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
        if self._yolo is not None:
            return self._yolo
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
            self._yolo = YOLO(self.model_path)
            # if ultralytics exposes .to(), try to move to device
            if hasattr(self._yolo, 'to') and getattr(self, 'device', None):
                try:
                    self._yolo.to(self.device)
                except Exception:
                    pass
            logger.info('YOLO model loaded from %s (device=%s)', os.path.abspath(self.model_path), getattr(self, 'device', 'unknown'))
        except Exception as e:
            logger.error('Erro ao carregar modelo YOLO: %s', e)
            raise
        return self._yolo

    def _load_vehicle_yolo(self):
        if self._vehicle_yolo is not None:
            return self._vehicle_yolo
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
            self._vehicle_yolo = YOLO(found)
            try:
                if hasattr(self._vehicle_yolo, 'to'):
                    self._vehicle_yolo.to(self.device)
            except Exception:
                pass
            logger.info('Loaded vehicle YOLO model %s', found)
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
        
        # 1. Hardcoded mapping para garantir que placas nunca falhem
        plate_synonyms = ['placa', 'license', 'plate', 'car_plate', '0']
        if any(syn in detected_name for syn in plate_synonyms):
            # Se você tiver labels diferentes para placa de moto e carro, 
            # pode refinar aqui, mas 'plate' é o padrão do seu XML
            return 'plate'

        # 2. Tentativa via config JSON (para veículos: car, truck, bus...)
        classes = self.classes_config.get('classes', [])
        for cls in classes:
            if detected_name == cls.get('id', '').lower() or detected_name == cls.get('label', '').lower():
                return cls.get('label')
            for kw in cls.get('keywords', []):
                if kw.lower() in detected_name: # 'in' é mais seguro que '=='
                    return cls.get('label')    
        return None
    
    def _determine_class_by_area(self, area_fraction):
        classes = self.classes_config.get('classes', [])
        if not classes:
            return 'object'
        selected = None
        best_ratio = -1.0
        for c in classes:
            r = c.get('min_area_ratio', 0)
            if r <= area_fraction and r > best_ratio:
                best_ratio = r
                selected = c
        if selected:
            return selected.get('label', selected.get('id', 'object'))
        fallback = min(classes, key=lambda x: x.get('min_area_ratio', 0))
        return fallback.get('label', fallback.get('id', 'object'))

    def _extract_coords(self, det, w, h):
        """Extrai coordenadas e converte para o formato 0-100 do Label Studio."""
        try:
            if hasattr(det, 'xywhr') and det.xywhr is not None: # Caso OBB
                cx, cy, bw, bh, rot_rad = det.xywhr.cpu().numpy().flatten().tolist()
                return {
                    'x': float(((cx - bw/2) / w) * 100.0),
                    'y': float(((cy - bh/2) / h) * 100.0),
                    'width': float((bw / w) * 100.0),
                    'height': float((bh / h) * 100.0),
                    'rotation': float(np.degrees(rot_rad))
                }
            elif hasattr(det, 'xyxy'): # Caso Box normal
                x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                return {
                    'x': float((x1 / w) * 100.0),
                    'y': float((y1 / h) * 100.0),
                    'width': float(((x2 - x1) / w) * 100.0),
                    'height': float(((y2 - y1) / h) * 100.0),
                    'rotation': 0.0
                }
        except:
            return None

    def _predict_image(self, image_url):
        img = load_image_from_url(image_url)
        w, h = img.size
        predictions_data = []

        # --- MODELO 1: PLACAS (YOLO Principal) ---
        yolo_plate = self._load_yolo()
        conf_p = float(os.getenv('YOLO_CONF', 0.15))
        # Rodamos o modelo de placa
        res_p_list = yolo_plate(img, conf=conf_p, verbose=False)
        
        for res_p in res_p_list:
            # Pegamos Boxes (BB) para evitar problemas com OBB
            if hasattr(res_p, 'boxes') and res_p.boxes is not None:
                for det in res_p.boxes:
                    cls_idx = int(det.cls.item())
                    label = self._map_label_by_name(res_p.names.get(cls_idx, "")) or "plate"
                    
                    # Extração BB padrão (xyxy)
                    x1, y1, x2, y2 = det.xyxy.cpu().numpy().flatten().tolist()
                    predictions_data.append({
                        'x': (x1 / w) * 100.0, 'y': (y1 / h) * 100.0,
                        'width': ((x2 - x1) / w) * 100.0, 'height': ((y2 - y1) / h) * 100.0,
                        'rotation': 0.0, 'score': float(det.conf.item()),
                        'label': label, 'orig_w': w, 'orig_h': h
                    })

        # --- MODELO 2: VEÍCULOS (Com Tiling para longa distância) ---
        try:
            v_yolo = self._load_vehicle_yolo()
            conf_v = float(os.getenv('YOLO_VEH_CONF', 0.25))
            
            # 1. Detecção Global (Normal)
            res_v_list = v_yolo(img, conf=conf_v, verbose=False)
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

            # 2. TILING: Se detectar poucos veículos, vasculha a imagem em blocos (Longa Distância)
            # Isso é o que o seu código antigo fazia para "enxergar" longe
            if len(predictions_data) < 5: 
                tiles = [
                    (0, 0, w//2, h//2), (w//2, 0, w, h//2),
                    (0, h//2, w//2, h), (w//2, h//2, w, h)
                ]
                for (tx1, ty1, tx2, ty2) in tiles:
                    tile_img = img.crop((tx1, ty1, tx2, ty2))
                    tile_res = v_yolo(tile_img, conf=conf_v + 0.1, verbose=False)
                    for r in tile_res:
                        if r.boxes:
                            for det in r.boxes:
                                label = self._map_label_by_name(r.names.get(int(det.cls.item()), ""))
                                if label and not label.endswith('plate'):
                                    bx1, by1, bx2, by2 = det.xyxy.cpu().numpy().flatten().tolist()
                                    # Converte coordenada do tile para coordenada global
                                    predictions_data.append({
                                        'x': ((bx1 + tx1) / w) * 100.0, 'y': ((by1 + ty1) / h) * 100.0,
                                        'width': ((bx2 - bx1) / w) * 100.0, 'height': ((by2 - by1) / h) * 100.0,
                                        'rotation': 0.0, 'score': float(det.conf.item()),
                                        'label': label, 'orig_w': w, 'orig_h': h
                                    })
        except Exception as e:
            logger.error(f"Erro veículos: {e}")

        return {'items': predictions_data, 'width': w, 'height': h}

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
        if not self.client:
            logger.error('Cliente Label Studio não está disponível')
            return
        
        project = self.client.get_project(project_id)
        tasks = self.get_unlabeled_tasks(project_id)
        
        for t in tasks:
            try:
                data = t.get('data') or {}
                raw_url = data.get('image') or data.get('image_url') or list(data.values())[0]
                image_url = get_absolute_url(raw_url)
                
                # Obtém as predições (OBB ou BB)
                pred = self._predict_image(image_url)
                
                # CORREÇÃO: Usamos 'items' conforme definido na função _predict_image
                predictions = self._generate_results(pred['items'])
                
                self._submit_predictions(t.get('id'), predictions, project_id)
            except Exception as e:
                logger.error('Erro ao processar tarefa %s: %s', t.get('id'), e)

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            try:
                data = task.get('data') or {}
                raw_url = data.get('image') or data.get('image_url') or list(data.values())[0]
                image_url = get_absolute_url(raw_url)
                
                # Obtém predições combinadas
                pred_data = self._predict_image(image_url)
                
                # O segredo: passamos a lista completa para o gerador de resultados
                results = self._generate_results(pred_data['items'])
                
                if results:
                    predictions.append(results[0])
                else:
                    predictions.append({'result': []})
            except Exception as e:
                logger.error(f'Erro na predição: {e}')
                predictions.append({'result': []})
        return predictions
        predictions = []
        for task in tasks:
            try:
                data = task.get('data') or {}
                raw_url = data.get('image') or data.get('image_url') or list(data.values())[0]
                image_url = get_absolute_url(raw_url)
                
                pred_data = self._predict_image(image_url)
                
                # Gera o formato final para todos os itens detectados
                formatted_preds = self._generate_results(pred_data['items'])
                
                if formatted_preds:
                    # formatted_preds[0] contém o dicionário {'result': [...]}
                    predictions.append(formatted_preds[0])
                else:
                    predictions.append({'result': []})
            except Exception as e:
                logger.error(f'Erro na predição: {e}')
                predictions.append({'result': []})
        return predictions
        """Método principal chamado pelo Label Studio para obter predições."""
        predictions = []
        for task in tasks:
            try:
                data = task.get('data') or {}
                # Tenta encontrar a URL da imagem em campos comuns
                raw_url = data.get('image') or data.get('image_url') or list(data.values())[0]
                image_url = get_absolute_url(raw_url)
                
                # 1. Obtém os dados brutos da detecção (veículos + placas)
                pred_data = self._predict_image(image_url)
                
                # 2. Converte para o formato JSON que o Label Studio entende
                # _generate_results retorna a lista formatada com 'result', 'score', etc.
                formatted_preds = self._generate_results(pred_data['items'])
                
                if formatted_preds:
                    predictions.append(formatted_preds[0])
                else:
                    predictions.append({'result': []})

            except Exception as e:
                logger.error(f'Erro na predição da tarefa {task.get("id")}: {e}')
                predictions.append({'result': []})
                
        return predictions
