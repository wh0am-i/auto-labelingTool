import os
import logging
import json
from uuid import uuid4
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import Client

logger = logging.getLogger(__name__)


def load_classes_config():
    candidates = [
        os.path.expanduser('../../../../sam2_classes.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sam2_classes.json'),
        os.path.join(os.path.dirname(__file__), 'sam2_classes.json'),
        os.path.abspath(os.path.join(os.getcwd(), 'sam2_classes.json')),
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
    logger.warning("Arquivo sam2_classes.json não encontrado. Usando fallback simples.")
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
    defined in `sam2_classes.json` using keyword matching and area heuristics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Client(url=os.getenv('LABEL_STUDIO_URL'), api_key=os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN')) if (os.getenv('LABEL_STUDIO_URL') and (os.getenv('LEGACY_TOKEN') or os.getenv('PERSONAL_TOKEN'))) else None
        self.classes_config = load_classes_config()
        self._yolo = None
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
        self._yolo = YOLO(self.model_path)
        logger.info('YOLO model loaded from %s', os.path.abspath(self.model_path))
        return self._yolo

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
        classes = self.classes_config.get('classes', [])
        
        # 1. Tentativa de match exato com o ID ou Label
        for cls in classes:
            if detected_name == cls.get('id').lower() or detected_name == cls.get('label').lower():
                return cls.get('label')
            
        for cls in classes:
            for kw in cls.get('keywords', []):
                # Se o nome detectado for EXATAMENTE a keyword
                if kw.lower() == detected_name:
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

    def _predict_image(self, image_url):
        img = load_image_from_url(image_url)
        w, h = img.size
        yolo = self._load_yolo()
        
        # Chamada para YOLOv11 (OBB)
        res = yolo(img, verbose=False)
        if isinstance(res, (list, tuple)):
            res = res[0]

        predictions_data = []
        try:
            # Verifica se existem detecções OBB primeiro
            has_obb = hasattr(res, 'obb') and res.obb is not None
            # Verifica se existem detecções de Boxes comuns
            has_boxes = hasattr(res, 'boxes') and res.boxes is not None
            
            items = []
            if has_obb and len(res.obb) > 0:
                items = res.obb
            elif has_boxes and len(res.boxes) > 0:
                items = res.boxes
            else:
                logger.info("Nenhuma detecção encontrada na imagem.")
                return {'items': [], 'width': w, 'height': h}
            
            for det in items:
                conf = float(det.conf.item())
                cls_idx = int(det.cls.item())
                name = res.names.get(cls_idx, str(cls_idx))
                
                label = self._map_label_by_name(name) or self._determine_class_by_area(0)

                # Extração para OBB (Oriented Bounding Box)
                if hasattr(det, 'xywhr') and det.xywhr is not None:
                    xywhr = det.xywhr.cpu().numpy().flatten().tolist()
                    if len(xywhr) == 5:
                        cx, cy, bw, bh, rotation_rad = xywhr
                        rotation_deg = float(np.degrees(rotation_rad))
                        x_pct = float(((cx - bw/2) / w) * 100.0)
                        y_pct = float(((cy - bh/2) / h) * 100.0)
                        width_pct = float((bw / w) * 100.0)
                        height_pct = float((bh / h) * 100.0)
                    else:
                        # Fallback se o formato xywhr for inesperado
                        continue
                else:
                    # Fallback para BB comum
                    xyxy = det.xyxy.cpu().numpy().flatten().tolist()
                    x1, y1, x2, y2 = xyxy
                    x_pct = float((x1 / w) * 100.0)
                    y_pct = float((y1 / h) * 100.0)
                    width_pct = float(((x2 - x1) / w) * 100.0)
                    height_pct = float(((y2 - y1) / h) * 100.0)
                    rotation_deg = 0.0

                predictions_data.append({
                    'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct,
                    'rotation': rotation_deg, 'score': conf, 'label': label,
                    'orig_w': w, 'orig_h': h
                })
        except Exception as e:
            logger.error('Erro ao interpretar resultados YOLO: %s', e)
            raise
        
        if has_obb and len(res.obb) > 0:
            logger.info(f"Detectadas {len(res.obb)} caixas OBB")
            items = res.obb
        elif has_boxes and len(res.boxes) > 0:
            logger.info(f"Detectadas {len(res.boxes)} caixas normais (BB)")
            items = res.boxes

        return {'items': predictions_data, 'width': w, 'height': h}
    
    def _generate_results(self, items):
        results = []
        total_score = 0.0
        
        for it in items:
            label = it['label']
            total_score += it['score']
            
            res_item = {
                'id': str(uuid4()),
                'from_name': label, 
                'to_name': 'image',
                'original_width': it['orig_w'],
                'original_height': it['orig_h'],
                'value': {
                    'x': it['x'],
                    'y': it['y'],
                    'width': it['width'],
                    'height': it['height'],
                    'rotation': it['rotation'], # Valor em graus
                    'rectanglelabels': [label]
                },
                'score': it['score'],
                'type': 'rectanglelabels',
                'readonly': False
            }
            results.append(res_item)
            
        avg = total_score / max(len(results), 1) if results else 0
        return [{'result': results, 'model_version': 'yolov11-obb', 'score': avg}]
        results = []
        total_score = 0.0
        
        for it in items:
            label = it['label']
            total_score += it['score']
            
            # CORREÇÃO CRUCIAL: 'from_name' deve ser o nome do Label no XML (car, bus, etc)
            # Como seu XML tem um RectangleLabels para cada classe, o from_name deve ser a própria classe.
            results.append({
                'id': str(uuid4()),
                'from_name': label, # Antes era 'label', agora bate com o XML
                'to_name': 'image',
                'original_width': it['orig_w'],
                'original_height': it['orig_h'],
                'value': {
                    'x': it['x'],
                    'y': it['y'],
                    'width': it['width'],
                    'height': it['height'],
                    'rotation': it['rotation'],
                    'rectanglelabels': [label]
                },
                'score': it['score'],
                'type': 'rectanglelabels',
                'readonly': False
            })
            
        avg = total_score / max(len(results), 1) if results else 0
        return [{'result': results, 'model_version': 'yolov11-obb', 'score': avg}]
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

    def predict(self, tasks):
        predictions = []
        for task in tasks:
            try:
                data = task.get('data') or {}
                raw_url = data.get('image') or data.get('image_url') or list(data.values())[0]
                image_url = get_absolute_url(raw_url)
                
                pred = self._predict_image(image_url)
                
                # CORREÇÃO: Usamos 'items' e estendemos a lista de resultados
                preds = self._generate_results(pred['items'])
                predictions.extend(preds)
            except Exception as e:
                logger.error('Erro na predição: %s', e)
        return predictions
