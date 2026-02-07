import os
import logging
import json
from uuid import uuid4
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import time
from datetime import datetime
try:
    import torch
except Exception:
    torch = None
try:
    import cv2
except Exception:
    cv2 = None

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
        # configurable thresholds (env vars) - defaults more permissive to detect small/multiple objects
        conf = float(os.getenv('YOLO_CONF', 0.10))
        iou = float(os.getenv('YOLO_IOU', 0.45))
        max_det = int(os.getenv('YOLO_MAX_DET', 300))

        # Chamada para YOLOv11 (OBB)
        try:
            res = yolo(img, verbose=False, conf=conf, iou=iou, max_det=max_det)
        except TypeError:
            # fallback if older ultralytics signature
            res = yolo(img, verbose=False)
        if isinstance(res, (list, tuple)):
            res = res[0]

        # Debug: quais atributos o resultado tem
        try:
            attrs = {k: getattr(res, k) is not None for k in ['boxes', 'obb', 'masks']}
            logger.debug(f'Result attributes presence: {attrs}')
        except Exception:
            logger.debug('Não foi possível inspecionar atributos de result')

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
                # tentar fallback a partir de máscaras (quando o modelo entrega masks)
                if hasattr(res, 'masks') and res.masks is not None and cv2 is not None:
                    try:
                        # tentar extrair masks em formato numpy (N,H,W)
                        masks_np = None
                        try:
                            masks_np = res.masks.data.cpu().numpy()
                        except Exception:
                            try:
                                masks_np = np.array(res.masks)
                            except Exception:
                                masks_np = None

                        if masks_np is not None and masks_np.ndim == 3:
                            logger.info('Fallback: gerando OBBs a partir de máscaras (%d masks)', masks_np.shape[0])
                            for mi in range(masks_np.shape[0]):
                                mask = (masks_np[mi].astype('uint8') * 255)
                                # encontra contornos
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if not contours:
                                    continue
                                largest = max(contours, key=cv2.contourArea)
                                if cv2.contourArea(largest) < 10:
                                    continue
                                rect = cv2.minAreaRect(largest)
                                ((cx, cy), (bw, bh), angle) = rect
                                # converter para porcentagens relativas
                                x_pct = float(((cx - bw/2) / w) * 100.0)
                                y_pct = float(((cy - bh/2) / h) * 100.0)
                                width_pct = float((bw / w) * 100.0)
                                height_pct = float((bh / h) * 100.0)
                                rotation_deg = float(angle)
                                # sem score/label conhecido; usar heurística de area
                                area_frac = (bw * bh) / (w * h)
                                label = self._determine_class_by_area(area_frac)
                                predictions_data.append({
                                    'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct,
                                    'rotation': rotation_deg, 'score': 0.5, 'label': label,
                                    'orig_w': w, 'orig_h': h
                                })
                            if predictions_data:
                                logger.info('Fallback OBBs gerados a partir de máscaras: %d', len(predictions_data))
                                # obter items a partir de predictions_data para geração posterior
                                return {'items': predictions_data, 'width': w, 'height': h}
                    except Exception as e:
                        logger.debug('Erro no fallback de máscaras->OBB: %s', e)

                logger.info("Nenhuma detecção encontrada na imagem.")
                return {'items': [], 'width': w, 'height': h}
            
            # debug: número de itens antes do parsing
            try:
                n_items = len(items)
            except Exception:
                try:
                    n_items = items.shape[0]
                except Exception:
                    n_items = -1
            logger.debug('Número bruto de detecções (items): %s', n_items)

            for idx, det in enumerate(items):
                # conf / cls leitura robusta
                try:
                    conf = float(det.conf.item())
                except Exception:
                    try:
                        conf = float(getattr(det, 'conf', 0.0))
                    except Exception:
                        conf = 0.0
                try:
                    cls_idx = int(det.cls.item())
                except Exception:
                    try:
                        cls_idx = int(getattr(det, 'cls', 0))
                    except Exception:
                        cls_idx = 0
                name = res.names.get(cls_idx, str(cls_idx))
                label = self._map_label_by_name(name) or self._determine_class_by_area(0)

                # Extração para OBB (Oriented Bounding Box)
                rotation_deg = 0.0
                x_pct = y_pct = width_pct = height_pct = 0.0
                try:
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
                            logger.debug('Formato xywhr inesperado: %s', xywhr)
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
                except Exception as e:
                    logger.debug('Erro extraindo coords de det %d: %s', idx, e)
                    continue

                logger.debug('Det %d: name=%s mapped_label=%s conf=%.4f bbox=[x=%.2f y=%.2f w=%.2f h=%.2f rot=%.2f]', idx, name, label, conf, x_pct, y_pct, width_pct, height_pct, rotation_deg)

                predictions_data.append({
                    'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct,
                    'rotation': rotation_deg, 'score': conf, 'label': label,
                    'orig_w': w, 'orig_h': h
                })

            logger.debug('Predictions_data length after parsing items: %d', len(predictions_data))
            # schedule async debug dump to avoid blocking main flow (controlled by DEBUG_DUMP env)
            try:
                if os.getenv('DEBUG_DUMP', '0') == '1':
                    t = threading.Thread(target=self._async_debug_dump, args=(image_url, img, res, predictions_data), daemon=True)
                    t.start()
            except Exception as e:
                logger.debug('Erro iniciando debug thread: %s', e)
            # Tiling fallback: se poucas detecções, tentar detectar em tiles (útil para placas pequenas/distantes)
            try:
                do_tiling = (len(predictions_data) < 3 and w > 800)
                if os.getenv('YOLO_FORCE_TILE', '0') == '1':
                    do_tiling = True
                if do_tiling:
                    logger.info('Executando detecção por tiles (fallback)')
                    tile_size = int(os.getenv('YOLO_TILE_SIZE', '800'))
                    overlap = float(os.getenv('YOLO_TILE_OVERLAP', '0.25'))
                    stride = int(tile_size * (1 - overlap))
                    img_arr = np.array(img)
                    all_boxes = []
                    all_scores = []
                    all_cls = []
                    for y0 in range(0, max(1, h - 1), max(1, stride)):
                        for x0 in range(0, max(1, w - 1), max(1, stride)):
                            x1 = x0 + tile_size
                            y1_ = y0 + tile_size
                            x2 = min(w, x1)
                            y2 = min(h, y1_)
                            # crop tile as PIL Image
                            tile = Image.fromarray(img_arr[y0:y2, x0:x2])
                            try:
                                tres = yolo(tile, verbose=False, conf=conf, iou=iou, max_det=max_det)
                            except TypeError:
                                tres = yolo(tile, verbose=False)
                            if isinstance(tres, (list, tuple)):
                                tres = tres[0]
                            if hasattr(tres, 'boxes') and tres.boxes is not None:
                                try:
                                    tb = tres.boxes.xyxy.cpu().numpy()
                                    tc = tres.boxes.cls.cpu().numpy()
                                    ts = tres.boxes.conf.cpu().numpy()
                                    for b, cls_idx, sc in zip(tb, tc, ts):
                                        # map to global coords
                                        bx1, by1, bx2, by2 = b.tolist()
                                        gx1 = bx1 + x0
                                        gy1 = by1 + y0
                                        gx2 = bx2 + x0
                                        gy2 = by2 + y0
                                        all_boxes.append([gx1, gy1, gx2, gy2])
                                        all_scores.append(float(sc))
                                        all_cls.append(int(cls_idx))
                                except Exception:
                                    pass
                    # apply NMS on aggregated boxes
                    if all_boxes:
                        keep_idx = self._nms_boxes(all_boxes, all_scores, iou_threshold=float(os.getenv('YOLO_TILE_NMS_IOU', '0.3')))
                        new_preds = []
                        img_arr_full = np.array(img)
                        for i in keep_idx:
                            bx1, by1, bx2, by2 = all_boxes[i]
                            sc = all_scores[i]
                            cls_idx = all_cls[i]
                            name = tres.names.get(cls_idx, str(cls_idx)) if 'tres' in locals() and hasattr(tres, 'names') else str(cls_idx)
                            label = self._map_label_by_name(name) or self._determine_class_by_area(((bx2-bx1)*(by2-by1))/(w*h) if w*h>0 else 0)
                            # attempt derive OBB from patch
                            obb = self._derive_obb_from_patch(img_arr_full, bx1, by1, bx2, by2)
                            if obb:
                                obb['score'] = sc
                                obb['label'] = label
                                obb['orig_w'] = w
                                obb['orig_h'] = h
                                new_preds.append(obb)
                            else:
                                x_pct = float((bx1 / w) * 100.0)
                                y_pct = float((by1 / h) * 100.0)
                                width_pct = float(((bx2 - bx1) / w) * 100.0)
                                height_pct = float(((by2 - by1) / h) * 100.0)
                                new_preds.append({'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct, 'rotation': 0.0, 'score': sc, 'label': label, 'orig_w': w, 'orig_h': h})
                        if new_preds:
                            logger.info('Tiles produced %d candidates, keeping %d after NMS', len(all_boxes), len(new_preds))
                            predictions_data = new_preds
            except Exception as e:
                logger.debug('Erro durante tiling fallback: %s', e)
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

    def _derive_obb_from_patch(self, img_arr, x1_px, y1_px, x2_px, y2_px):
        """Try derive oriented bbox from a cropped patch using contours/minAreaRect."""
        if cv2 is None:
            return None
        try:
            img_h, img_w = img_arr.shape[:2]
            x1_px = max(0, int(x1_px))
            y1_px = max(0, int(y1_px))
            x2_px = min(img_w - 1, int(x2_px))
            y2_px = min(img_h - 1, int(y2_px))
            if x2_px <= x1_px or y2_px <= y1_px:
                return None
            patch = img_arr[y1_px:y2_px, x1_px:x2_px]
            if patch.size == 0:
                return None
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                edges = cv2.Canny(blur, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return None
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 20:
                return None
            rect = cv2.minAreaRect(largest)
            # rect: ((cx,cy),(bw,bh), angle)
            (cx_p, cy_p), (bw_px, bh_px), angle = rect
            # translate center to global coords
            cx = x1_px + cx_p
            cy = y1_px + cy_p
            # normalize angle/size: OpenCV angle convention requires normalization
            # ensure bw_px is width along the rotated box reference
            if bw_px < bh_px:
                # swap to keep bw_px >= bh_px and adjust angle
                bw_px, bh_px = bh_px, bw_px
                angle = angle + 90.0
            # normalize angle to [-180,180)
            while angle >= 180.0:
                angle -= 360.0
            while angle < -180.0:
                angle += 360.0

            # compute top-left from center (consistent with upstream xywhr handling)
            if img_w == 0 or img_h == 0:
                return None
            x_pct = float(((cx - bw_px / 2.0) / img_w) * 100.0)
            y_pct = float(((cy - bh_px / 2.0) / img_h) * 100.0)
            width_pct = float((bw_px / img_w) * 100.0)
            height_pct = float((bh_px / img_h) * 100.0)
            rotation_deg = float(angle)
            return {'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct, 'rotation': rotation_deg}
        except Exception:
            return None
    
    def _async_debug_dump(self, image_url, img, res, predictions_data):
        try:
            debug_root = os.path.join(os.path.dirname(__file__), 'debug')
            os.makedirs(debug_root, exist_ok=True)

            # maintain a run counter to create a subfolder per execution
            counter_file = os.path.join(debug_root, 'run_counter.txt')
            try:
                if os.path.exists(counter_file):
                    with open(counter_file, 'r') as cf:
                        cnt = int(cf.read().strip() or '0')
                else:
                    cnt = 0
            except Exception:
                cnt = 0
            cnt += 1
            try:
                with open(counter_file + '.tmp', 'w') as cf:
                    cf.write(str(cnt))
                os.replace(counter_file + '.tmp', counter_file)
            except Exception:
                try:
                    with open(counter_file, 'w') as cf:
                        cf.write(str(cnt))
                except Exception:
                    pass

            run_name = f"run_{cnt:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir = os.path.join(debug_root, run_name)
            os.makedirs(run_dir, exist_ok=True)

            try:
                base_name = os.path.splitext(os.path.basename(image_url))[0]
            except Exception:
                base_name = 'img'
            uid = str(uuid4())[:8]
            debug_base = f"{base_name}_{uid}"

            dump = {'predictions_data': predictions_data}

            # Safely try to read raw attributes
            try:
                if hasattr(res, 'boxes') and getattr(res, 'boxes') is not None:
                    try:
                        bx = getattr(res.boxes, 'xyxy', None)
                        if bx is not None:
                            dump['raw_boxes_xyxy'] = bx.cpu().numpy().tolist()
                        conf = getattr(res.boxes, 'conf', None)
                        if conf is not None:
                            dump['raw_boxes_conf'] = conf.cpu().numpy().tolist()
                        cls = getattr(res.boxes, 'cls', None)
                        if cls is not None:
                            dump['raw_boxes_cls'] = cls.cpu().numpy().tolist()
                    except Exception:
                        dump['raw_boxes_repr'] = str(res.boxes)
                if hasattr(res, 'obb') and getattr(res, 'obb') is not None:
                    try:
                        dump['raw_obb'] = []
                        for o in res.obb:
                            try:
                                dump['raw_obb'].append(o.cpu().numpy().tolist())
                            except Exception:
                                dump['raw_obb'].append(str(o))
                    except Exception:
                        dump['raw_obb_repr'] = str(res.obb)
                if hasattr(res, 'masks') and getattr(res, 'masks') is not None:
                    try:
                        m = getattr(res.masks, 'data', None)
                        if m is not None:
                            dump['masks_shape'] = m.cpu().numpy().shape
                        else:
                            dump['masks_repr'] = str(res.masks)
                    except Exception:
                        dump['masks_error'] = 'failed to read masks'
            except Exception:
                dump['raw_error'] = 'failed to introspect res'

            # atomic json write
            json_path = os.path.join(run_dir, debug_base + '.json')
            tmp_json = json_path + '.tmp'
            try:
                with open(tmp_json, 'w', encoding='utf8') as jf:
                    json.dump(dump, jf, indent=2, ensure_ascii=False)
                os.replace(tmp_json, json_path)
            except Exception:
                try:
                    if os.path.exists(tmp_json):
                        os.remove(tmp_json)
                except Exception:
                    pass

            # visualization (best-effort)
            try:
                vis_path = os.path.join(run_dir, debug_base + '.png')
                tmp_vis = vis_path + '.tmp.png'
                vis_img = np.array(img).copy()
                if cv2 is not None:
                    for pd in predictions_data:
                        try:
                            x_pct = float(pd.get('x', 0))
                            y_pct = float(pd.get('y', 0))
                            w_pct = float(pd.get('width', 0))
                            h_pct = float(pd.get('height', 0))
                            rot = float(pd.get('rotation', 0) or 0)
                            score = float(pd.get('score', 0) or 0)
                            lbl = str(pd.get('label', ''))
                        except Exception:
                            continue
                        ih, iw = vis_img.shape[0], vis_img.shape[1]
                        x_px = int((x_pct/100.0) * iw)
                        y_px = int((y_pct/100.0) * ih)
                        bw_px = int((w_pct/100.0) * iw)
                        bh_px = int((h_pct/100.0) * ih)
                        if bw_px <= 0 or bh_px <= 0:
                            continue
                        if abs(rot) > 0.1:
                            rect = ((x_px + bw_px/2, y_px + bh_px/2), (bw_px, bh_px), rot)
                            try:
                                box = cv2.boxPoints(rect).astype(int)
                                cv2.drawContours(vis_img, [box], 0, (0,255,0), 2)
                                cv2.putText(vis_img, f"{lbl}:{score:.2f}", (max(0,box[0][0]), max(0,box[0][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            except Exception:
                                pass
                        else:
                            x2 = x_px + bw_px
                            y2 = y_px + bh_px
                            try:
                                cv2.rectangle(vis_img, (x_px, y_px), (x2, y2), (0,255,0), 2)
                                cv2.putText(vis_img, f"{lbl}:{score:.2f}", (x_px, max(0,y_px-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                            except Exception:
                                pass
                    cv2.imwrite(tmp_vis, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    os.replace(tmp_vis, vis_path)
                else:
                    # fallback: save raw image via PIL
                    try:
                        Image.fromarray(vis_img).save(tmp_vis, format='PNG')
                        os.replace(tmp_vis, vis_path)
                    except Exception:
                        # final fallback: save without tmp
                        try:
                            Image.fromarray(vis_img).save(vis_path, format='PNG')
                        except Exception:
                            pass
            except Exception:
                logger.debug('Erro salvando visualizacao debug', exc_info=True)

        except Exception:
            logger.debug('Erro em _async_debug_dump', exc_info=True)

    def _generate_results(self, items):
        results = []
        total_score = 0.0
        
        for it in items:
            label = it['label']
            total_score += it['score']
            # `from_name` must match the RectangleLabels control name in your label config
            control_name = 'label'  # your XML uses name="label" for RectangleLabels
            res_item = {
                'id': str(uuid4()),
                'from_name': control_name,
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
