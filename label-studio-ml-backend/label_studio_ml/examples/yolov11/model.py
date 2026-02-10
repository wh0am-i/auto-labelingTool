"""
Backward-compatible YOLOv11 backend for vehicle detection (yolo11x).

This module provides a `NewModel` that uses a yolo11x checkpoint by
default (examples/yolov11/models/yolo11x.pt) and maps detections to
classes defined in `sam2_classes.json` (vehicles, trucks, buses, etc.).
"""

def load_classes_config():
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'sam2_classes.json')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sam2_classes.json')),
        os.path.join(os.path.dirname(__file__), 'sam2_classes.json'),
        os.path.abspath(os.path.join(os.getcwd(), 'sam2_classes.json')),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf8') as f:
                    return json.load(f)
            except Exception:
                continue
    return {"classes": []}


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
        # Resolve YOLO model path: prefer explicit env var, else use yolo11/models/yolo11x.pt
        env_path = os.getenv('YOLO_MODEL_PATH')
        candidates = []
        if env_path:
            candidates.append(os.path.abspath(os.path.expanduser(env_path)))
        # default for vehicle detector (yolo11x)
        candidates.append(os.path.join(os.path.dirname(__file__), 'models', os.path.basename(env_path or 'yolo11x.pt')))
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
        # configurable thresholds (env vars) - make defaults more permissive to detect small/multiple objects
        conf = float(os.getenv('YOLO_CONF', 0.08))
        iou = float(os.getenv('YOLO_IOU', 0.60))
        max_det = int(os.getenv('YOLO_MAX_DET', 500))

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

        # Debug: inspecionar shapes/arrays de boxes/masks quando disponíveis
        try:
            if hasattr(res, 'boxes') and res.boxes is not None:
                try:
                    xy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy()
                    logger.debug('res.boxes present: %s boxes; confs(top5)=%s; cls(top5)=%s', xy.shape[0], confs[:5].tolist() if confs.size>0 else [], cls[:5].tolist() if cls.size>0 else [])
                except Exception as e:
                    logger.debug('res.boxes present but failed to read arrays: %s', e)
            if hasattr(res, 'obb') and res.obb is not None:
                try:
                    obb_len = len(res.obb)
                    logger.debug('res.obb present: %d', obb_len)
                except Exception:
                    logger.debug('res.obb present (len unreadable)')
            if hasattr(res, 'masks') and res.masks is not None:
                logger.debug('res.masks present')
        except Exception:
            pass

        # Simplified: usar apenas boxes.xyxy e aceitar somente classes mapeadas
        predictions_data = []
        try:
            if not (hasattr(res, 'boxes') and res.boxes is not None):
                logger.info('Nenhuma detecção em boxes (xyxy) encontrada.')
                return {'items': [], 'width': w, 'height': h}
            xyxy_arr = getattr(res.boxes, 'xyxy', None)
            conf_arr = getattr(res.boxes, 'conf', None)
            cls_arr = getattr(res.boxes, 'cls', None)
            if xyxy_arr is None:
                return {'items': [], 'width': w, 'height': h}
            xyxy = xyxy_arr.cpu().numpy()
            confs = conf_arr.cpu().numpy() if conf_arr is not None else np.zeros((len(xyxy),), dtype=float)
            clss = cls_arr.cpu().numpy() if cls_arr is not None else np.zeros((len(xyxy),), dtype=int)
            for b, sc, cls_idx in zip(xyxy, confs, clss):
                try:
                    bx1, by1, bx2, by2 = [float(x) for x in b.tolist()]
                except Exception:
                    continue
                name = res.names.get(int(cls_idx), str(int(cls_idx))) if hasattr(res, 'names') else str(int(cls_idx))
                label = self._map_label_by_name(name)
                if label is None:
                    logger.debug('Descartando detecção não mapeada: %s', name)
                    continue
                x_pct = float((bx1 / w) * 100.0)
                y_pct = float((by1 / h) * 100.0)
                width_pct = float(((bx2 - bx1) / w) * 100.0)
                height_pct = float(((by2 - by1) / h) * 100.0)
                predictions_data.append({'x': x_pct, 'y': y_pct, 'width': width_pct, 'height': height_pct, 'rotation': 0.0, 'score': float(sc), 'label': label, 'orig_w': w, 'orig_h': h})
        except Exception as e:
            logger.error('Erro ao interpretar resultados YOLO: %s', e)
            raise
        return {'items': predictions_data, 'width': w, 'height': h}
    
    def _generate_results(self, items):
        results = []
        total_score = 0.0

        for it in items:
            try:
                score = float(it.get('score', 0.0))
            except Exception:
                score = 0.0
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

        avg = total_score / len(results) if results else 0
        return [{'result': results, 'model_version': 'yolov11-bb', 'score': avg}]
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
