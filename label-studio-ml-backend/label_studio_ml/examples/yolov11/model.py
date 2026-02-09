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
                    
                        # Tentativa de derivar OBB a partir do patch da caixa (Canny -> minAreaRect)
                        try:
                            if cv2 is not None:
                                # crop patch in pixel coords
                                x1_px = max(0, int(round(x1)))
                                y1_px = max(0, int(round(y1)))
                                x2_px = min(w, int(round(x2)))
                                y2_px = min(h, int(round(y2)))
                                if x2_px > x1_px and y2_px > y1_px:
                                    arr = np.array(img)
                                    patch = arr[y1_px:y2_px, x1_px:x2_px]
                                    if patch.size != 0:
                                        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                                        blur = cv2.GaussianBlur(gray, (5,5), 0)
                                        edges = cv2.Canny(blur, 50, 150)
                                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        if contours:
                                            largest = max(contours, key=cv2.contourArea)
                                            if cv2.contourArea(largest) > 10:
                                                rect = cv2.minAreaRect(largest)
                                                ((cx_p, cy_p), (bw_px, bh_px), angle) = rect
                                                # convert to global coords
                                                cx = x1_px + cx_p
                                                cy = y1_px + cy_p
                                                rotation_deg = float(angle)
                                                width_pct = float((bw_px / w) * 100.0)
                                                height_pct = float((bh_px / h) * 100.0)
                                                x_pct = float(((cx - bw_px/2) / w) * 100.0)
                                                y_pct = float(((cy - bh_px/2) / h) * 100.0)
                                                logger.debug('OBB derivado do patch: angle=%.2f bw=%d bh=%d', rotation_deg, int(bw_px), int(bh_px))
                        except Exception as e:
                            logger.debug('Erro derivando OBB do patch: %s', e)
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
