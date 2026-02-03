import os
import logging
import json
from pathlib import Path
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk import Client
from label_studio_sdk.converter import brush
from uuid import uuid4
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import cv2
import re

# Carrega .env automaticamente se existir (caminho do projeto)
def load_env_paths():
    candidates = [
        os.path.expanduser('~/Documentos/labelStudio/.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'),
        os.path.join(os.path.dirname(__file__), '.env'),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            try:
                with open(p, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or line.startswith('//'):
                            continue
                        if '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('\'"')
                        if k not in os.environ:
                            os.environ[k] = v
                logger.info(f"Loaded env from {p}")
                return
            except Exception:
                continue

# carregar antes de ler LABEL_STUDIO_API_KEY
load_env_paths()

# Configuração de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve paths relative to the examples directory (one level up from this plugin)
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))                         # .../examples/segment_anything_2_image
EXAMPLES_DIR = os.path.abspath(os.path.join(PLUGIN_DIR, ".."))                  # .../examples

def find_existing_path(*candidates):
    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    return None

# Configuração do cliente Label Studio
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY') or os.getenv('API_KEY')
logger.info(f"Chave capturada do env")
LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL', 'http://127.0.0.1:8080')

try:
    client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    logger.info(f"Conectado ao Label Studio em {LABEL_STUDIO_URL}")
except Exception as e:
    logger.warning(f"Aviso ao conectar ao Label Studio: {e}")
    client = None

# SAM2.1: use the format WITHOUT the directory prefix
# The SAM2 package handles hydra config discovery internally
DEFAULT_CONFIG_NAME = os.getenv('MODEL_CONFIG', 'sam2.1/sam2.1_hiera_l')
DEFAULT_CHECKPOINT_NAME = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')
DEVICE = os.getenv('DEVICE', 'cpu')
logger.info("Utilizando device %s (configure DEVICE env var para 'cuda' se GPU disponível).", DEVICE)

# Checkpoint candidates (absolute paths)
candidates_ckpt = [
    os.getenv('MODEL_CHECKPOINT') if os.path.isabs(os.getenv('MODEL_CHECKPOINT', '')) else None,
    os.path.join(EXAMPLES_DIR, 'checkpoints', DEFAULT_CHECKPOINT_NAME),
    os.path.join(PLUGIN_DIR, 'checkpoints', DEFAULT_CHECKPOINT_NAME),
    os.path.join(EXAMPLES_DIR, DEFAULT_CHECKPOINT_NAME),
]

RESOLVED_MODEL_CHECKPOINT = find_existing_path(*[c for c in candidates_ckpt if c])

# Carrega configuração de classes
def load_classes_config():
    """Carrega configuração de classes do arquivo sam2_classes.json"""
    candidates = [
        os.path.expanduser('~/Documentos/labelStudio/sam2_classes.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sam2_classes.json'),
        os.path.join(os.path.dirname(__file__), 'sam2_classes.json'),
    ]
    
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            try:
                with open(p, 'r') as f:
                    config = json.load(f)
                logger.info(f"Carregado arquivo de classes: {p}")
                return config
            except Exception as e:
                logger.warning(f"Erro ao carregar {p}: {e}")
    
    logger.warning("Arquivo sam2_classes.json não encontrado. Usando todas as máscaras como 'segmentation'.")
    return {"classes": [], "filtering_rules": {"enabled": False}}

CLASSES_CONFIG = load_classes_config()

# Variáveis globais para lazy loading
sam2_model = None
predictor = None


def initialize_sam2():
    """Inicializa o modelo SAM2.1 sob demanda
    
    Tenta registrar o diretório de configs do pacote `sam2` no Hydra
    usando vários `version_base` (compatibilidade com diferentes versões do Hydra).
    Se a composição do config for bem sucedida, build_sam2 é chamado.
    """
    global sam2_model, predictor

    if sam2_model is not None:
        return predictor

    try:
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from pathlib import Path

        checkpoint_path = RESOLVED_MODEL_CHECKPOINT or os.path.join(EXAMPLES_DIR, 'checkpoints', DEFAULT_CHECKPOINT_NAME)

        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint não encontrado: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

        sam2_dir = Path(sam2.__file__).parent
        config_dir = sam2_dir / "configs"
        logger.info(f"SAM2 configs dir: {config_dir}")

        # Try multiple hydra version_base options and config-name formats
        version_bases = [None, "1.2", "1.1", "1.0"]
        config_name_candidates = [DEFAULT_CONFIG_NAME, DEFAULT_CONFIG_NAME.replace("sam2.1/", "sam2.1/"), DEFAULT_CONFIG_NAME.replace("/", "_"), DEFAULT_CONFIG_NAME.split('/')[-1]]

        last_exc = None
        for vb in version_bases:
            try:
                GlobalHydra.instance().clear()
                logger.debug(f"Trying Hydra initialize_config_dir version_base={vb}")
                with initialize_config_dir(config_dir=str(config_dir), version_base=vb):
                    for cname in config_name_candidates:
                        try:
                            logger.debug(f"Trying compose config_name={cname}")
                            cfg = compose(config_name=cname)
                            logger.info(f"Hydra compose succeeded with config_name={cname} version_base={vb}")
                            # build_sam2 should now be able to resolve config
                            sam2_model = build_sam2(cname, checkpoint_path, device=DEVICE)
                            predictor = SAM2ImagePredictor(sam2_model)
                            logger.info("✓ Modelo SAM2.1 carregado com sucesso!")
                            return predictor
                        except Exception as e_inner:
                            logger.debug(f"compose/build attempt failed for {cname} (vb={vb}): {e_inner}")
                            last_exc = e_inner
            except Exception as e_vb:
                logger.debug(f"Hydra initialize_config_dir failed for version_base={vb}: {e_vb}")
                last_exc = e_vb
                continue

        # If we reach here, none of the attempts worked
        logger.error("Não foi possível inicializar SAM2.1 após várias tentativas.")
        if last_exc:
            raise last_exc
        else:
            raise RuntimeError("Unknown error initializing SAM2.1")

    except Exception as e:
        logger.error(f"Erro ao inicializar SAM2.1: {e}")
        logger.info("Verifique:\n - se o pacote `sam2` instalado contém a pasta `configs`\n - se o nome do config está correto (ex.: 'sam2.1/sam2.1_hiera_l')\n - compatibilidade entre versão do Hydra e do pacote sam2")
        raise


class NewModel(LabelStudioMLBase):
    """Modelo de Backend Customizado para Auto-Labeling com SAM2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.predictor = None
        self.classes_config = CLASSES_CONFIG

    def list_projects(self):
        """Lista todos os projetos disponíveis no Label Studio"""
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return []
        
        try:
            projects = self.client.get_projects()
            logger.info(f"Projetos encontrados: {len(projects)}")
            return projects
        except Exception as e:
            logger.error(f"Erro ao listar projetos: {e}")
            return []

    def get_unlabeled_tasks(self, project_id):
        """Obtém todas as imagens não etiquetadas de um projeto"""
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return []

        try:
            project = self.client.get_project(project_id)
            tasks = project.get_tasks()

            # Task is a dict: check if it has annotations or is_labeled flag
            unlabeled_tasks = [
                task for task in tasks
                if not task.get("annotations") or task.get("is_labeled") == False
            ]

            logger.info(
                f"Imagens não etiquetadas no projeto {project_id}: {len(unlabeled_tasks)}"
            )
            return unlabeled_tasks

        except Exception as e:
            logger.error(f"Erro ao buscar tarefas não etiquetadas: {e}")
            return []

    def auto_label_project(self, project_id):
        """Realiza auto-labeling em todas as imagens não etiquetadas do projeto"""
        logger.info(f"Iniciando auto-labeling do projeto {project_id}")
        logger.info(f"Classes configuradas: {[c['id'] for c in self.classes_config.get('classes', [])]}")

        try:
            self.predictor = initialize_sam2()
        except Exception as e:
            logger.error(f"Não foi possível inicializar SAM2: {e}")
            raise

        project = self.client.get_project(project_id)

        # Extrai labels do projeto e mescla com configuração local (mantendo min_area_ratio se existir)
        try:
            project_labels = self._extract_labels_from_project(project)
            if project_labels:
                local_classes = {c.get("id", c.get("label")): c for c in self.classes_config.get("classes", [])}
                merged = []
                for lbl in project_labels:
                    if lbl in local_classes:
                        merged.append(local_classes[lbl])
                    else:
                        merged.append({"id": lbl, "label": lbl, "min_area_ratio": 0})
                self.classes_config["classes"] = merged
                logger.info(f"Classes atualizadas (do projeto): {[c['id'] for c in merged]}")
        except Exception as e:
            logger.warning(f"Não foi possível extrair classes do projeto: {e}")

        unlabeled_tasks = self.get_unlabeled_tasks(project_id)

        if not unlabeled_tasks:
            logger.info("Nenhuma tarefa não etiquetada encontrada")
            return

        logger.info(f"Processando {len(unlabeled_tasks)} tarefas")

        for idx, task in enumerate(unlabeled_tasks):
            try:
                task_id = task["id"]
                task_data = task.get("data", {})

                logger.info(
                    f"Processando tarefa {idx + 1}/{len(unlabeled_tasks)} (ID: {task_id})"
                )

                img_url = task_data.get("image")
                if not img_url:
                    logger.warning(f"Tarefa {task_id} não contém URL de imagem")
                    continue

                predictor_results = self._sam_predict(
                    img_url=img_url,
                    task_id=task_id
                )

                from_name, to_name, value, result_type = self._get_first_tag_occurence(project)

                results = self._get_results(
                    masks=predictor_results["masks"],
                    probs=predictor_results["probs"],
                    width=predictor_results["width"],
                    height=predictor_results["height"],
                    from_name=from_name,
                    to_name=to_name,
                    class_labels=predictor_results.get("class_labels", ["segmentation"]),
                    result_type=result_type
                )

                # pass project_id so we can use Project API to create prediction
                self._submit_predictions(task_id, results, project_id)
                logger.info(f"Auto-labeling concluído para a tarefa {task_id}")

            except Exception as e:
                logger.error(f"Erro ao processar tarefa {task.get('id')}: {e}")
                continue

    def predict(self, tasks):
        """Predição para backend do Label Studio"""
        predictions = []

        try:
            self.predictor = initialize_sam2()
        except Exception as e:
            logger.error(f"Não foi possível inicializar SAM2: {e}")
            return predictions

        for task in tasks:
            try:
                task_id = task["id"]
                task_data = task.get("data", {})

                img_url = task_data.get("image")
                if not img_url:
                    continue

                predictor_results = self._sam_predict(
                    img_url=img_url,
                    task_id=task_id
                )

                project = self.client.get_project(task.get("project"))
                from_name, to_name, value, result_type = self._get_first_tag_occurence(project)

                results = self._get_results(
                    masks=predictor_results["masks"],
                    probs=predictor_results["probs"],
                    width=predictor_results["width"],
                    height=predictor_results["height"],
                    from_name=from_name,
                    to_name=to_name,
                    label="segmentation",
                    result_type=result_type
                )

                predictions.append({
                    "task_id": task_id,
                    "predictions": results
                })

            except Exception as e:
                logger.error(f"Erro ao processar tarefa {task.get('id')}: {e}")
                continue

        return predictions

    def _get_first_tag_occurence(self, project):
        """Extrai from_name e to_name do label_config do projeto e determina result_type.

        Busca por:
         - from_name: atributo name em <Labels>, <BrushLabels>, <RectangleLabels>
         - to_name: atributo name em <Image>
        Faz fallback para nomes padrão se não encontrados.
        """
        try:
            cfg = getattr(project, "label_config", "") or ""
            # procura 'from' em Labels/BrushLabels/RectangleLabels com atributo name="..."
            m = re.search(r'<\s*(BrushLabels|RectangleLabels|Labels)[^>]*\bname\s*=\s*["\']([^"\']+)["\']', cfg, flags=re.IGNORECASE)
            from_name = m.group(2) if m else 'label'

            # procura 'to' em Image com atributo name="..."
            m2 = re.search(r'<\s*Image[^>]*\bname\s*=\s*["\']([^"\']+)["\']', cfg, flags=re.IGNORECASE)
            to_name = m2.group(1) if m2 else 'image'

            # determina resultado: brush se houver BrushLabels, rectangle se RectangleLabels
            if re.search(r'<\s*BrushLabels\b', cfg, flags=re.IGNORECASE):
                result_type = 'brush'
            elif re.search(r'<\s*RectangleLabels\b', cfg, flags=re.IGNORECASE):
                result_type = 'rectangle'
            else:
                result_type = 'brush'

            logger.info(f"Tags extraídos: from_name={from_name}, to_name={to_name}, result_type={result_type}")
            return from_name, to_name, 'image', result_type
        except Exception as e:
            logger.warning(f"Erro ao extrair tags: {e}, usando padrão")
            return 'label', 'image', 'image', 'brush'


    def _generate_results(self, masks, scores, width, height, from_name, to_name, class_labels):
        """Generate prediction results for Label Studio.

        - Masks are encoded in RLE format.
        - Each mask is associated with a class label and score.
        """
        results = []
        total_score = 0

        for idx, (mask, score, class_label) in enumerate(zip(masks, scores, class_labels)):
            mask_binary = (mask * 255).astype(np.uint8)
            rle_encoded = brush.mask2rle(mask_binary)
            total_score += score

            logger.debug(f"Result {idx + 1}: Class='{class_label}', Score={score:.4f}")

            results.append({
                "id": str(uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "original_width": width,
                "original_height": height,
                "value": {
                    "format": "rle",
                    "rle": rle_encoded,
                    "labels": [class_label],  # Use "labels" instead of "brushlabels"
                },
                "score": float(score),
                "type": "segmentation",  # Use a more descriptive type
                "readonly": False,
            })

        logger.info(f"Generated {len(results)} prediction results.")
        return {
            "results": results,
            "average_score": total_score / max(len(results), 1) if results else 0,
        }


    def _predict_masks(self, image_url, task_id=None):
        """Perform prediction using the SAM2 model and filter results based on configuration."""
        try:
            image = self._load_image(image_url, task_id)
            height, width = image.shape[:2]
            image_area = height * width

            # Generate predictions using the SAM2 model
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
            )

            # Sort masks by scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            masks = masks[sorted_indices]
            scores = scores[sorted_indices]

            # Normalize masks to [0, 1] range
            normalized_masks = masks.astype(np.float32) / 255.0 if masks.max() > 1 else masks.astype(np.float32)

            # Filter masks and assign class labels
            filtered_masks, filtered_scores, filtered_labels = [], [], []
            filtering_rules = self.classes_config.get("filtering_rules", {})
            min_area_pct = filtering_rules.get("min_mask_area_percentage", 0.05)
            max_area_pct = filtering_rules.get("max_mask_area_percentage", 95.0)

            for mask, score in zip(normalized_masks, scores):
                mask_area_fraction = mask.sum() / image_area
                mask_area_percentage = mask_area_fraction * 100.0

                if filtering_rules.get("enabled", False):
                    if not (min_area_pct <= mask_area_percentage <= max_area_pct):
                        logger.debug(f"Mask filtered out: Area={mask_area_percentage:.2f}%")
                        continue

                class_label = self._determine_class_by_area(mask_area_fraction)
                logger.debug(f"Mask accepted: Area={mask_area_percentage:.2f}%, Class='{class_label}'")

                filtered_masks.append(mask)
                filtered_scores.append(score)
                filtered_labels.append(class_label)

            if not filtered_masks:
                logger.warning("No masks passed the filtering rules. Returning all masks.")
                filtered_masks = normalized_masks
                filtered_scores = scores
                filtered_labels = [
                    self._determine_class_by_area(mask.sum() / image_area) for mask in normalized_masks
                ]

            logger.info(f"Prediction completed: {len(filtered_masks)} masks retained.")
            return {
                "masks": filtered_masks,
                "scores": filtered_scores,
                "width": width,
                "height": height,
                "class_labels": filtered_labels,
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def _assign_label_by_size(self, area_frac):
        """Heurística simples: escolhe classe baseada na área da máscara (mask_area / image_area) usando sam2_classes.json.
        Retorna o rótulo da classe correspondente.
        """
        try:
            classes = self.classes_config.get("classes", [])
            if not classes:
                logger.warning("Nenhuma classe configurada em sam2_classes.json. Usando 'segmentation' como padrão.")
                return "segmentation"

            # Escolher a classe com o maior min_area_ratio <= area_frac
            best = None
            best_ratio = -1.0
            for c in classes:
                min_ratio = float(c.get("min_area_ratio", 0))
                if area_frac >= min_ratio and min_ratio > best_ratio:
                    best = c
                    best_ratio = min_ratio

            if best:
                label = best.get("label", best.get("id", "segmentation"))
                logger.debug(f"Classe '{label}' atribuída para área {area_frac:.4f}")
                return label

            # Fallback para a classe com o menor min_area_ratio
            classes_sorted = sorted(classes, key=lambda x: x.get("min_area_ratio", 0))
            fallback_label = classes_sorted[0].get("label", classes_sorted[0].get("id", "segmentation"))
            logger.debug(f"Fallback: Classe '{fallback_label}' atribuída para área {area_frac:.4f}")
            return fallback_label

        except Exception as e:
            logger.error(f"Erro ao atribuir classe: {e}")
            return "segmentation"


    def _determine_class_by_area(self, area_fraction):
        """Determine the class label based on the mask area fraction."""
        try:
            classes = self.classes_config.get("classes", [])
            if not classes:
                logger.warning("No classes configured. Defaulting to 'segmentation'.")
                return "segmentation"

            # Find the class with the largest min_area_ratio <= area_fraction
            selected_class = None
            highest_ratio = -1.0
            for cls in classes:
                min_ratio = float(cls.get("min_area_ratio", 0))
                if area_fraction >= min_ratio > highest_ratio:
                    selected_class = cls
                    highest_ratio = min_ratio

            if selected_class:
                label = selected_class.get("label", selected_class.get("id", "segmentation"))
                logger.debug(f"Class '{label}' selected for area fraction {area_fraction:.4f}")
                return label

            # Fallback to the class with the smallest min_area_ratio
            fallback_class = min(classes, key=lambda cls: cls.get("min_area_ratio", 0))
            fallback_label = fallback_class.get("label", fallback_class.get("id", "segmentation"))
            logger.debug(f"Fallback class '{fallback_label}' selected for area fraction {area_fraction:.4f}")
            return fallback_label

        except Exception as e:
            logger.error(f"Error determining class: {e}")
            return "segmentation"


    def _get_results(self, masks, probs, width, height, from_name, to_name, class_labels, result_type='brush'):
        """Gera os resultados de predição para o Label Studio.

        - Retorna as máscaras no formato RLE (segmentação).
        """
        results = []
        total_prob = 0

        for mask, prob, class_label in zip(masks, probs, class_labels):
            label_id = str(uuid4())[:4]
            mask_uint8 = (mask * 255).astype(np.uint8)
            rle = brush.mask2rle(mask_uint8)
            total_prob += prob

            logger.debug(f"Gerando predição: Classe='{class_label}', Probabilidade={prob:.4f}")

            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [class_label],  # Atribuir a classe correta
                },
                'score': float(prob),
                'type': 'brushlabels',
                'readonly': False
            })

        logger.info(f"Total de predições geradas: {len(results)}")
        return [{
            'result': results,
            'model_version': 'sam2-auto-label',
            'score': total_prob / max(len(results), 1) if results else 0
        }]


    def _extract_labels_from_project(self, project):
        """Extrai dinamicamente labels definidos no label_config do projeto Label Studio.
        Filtra labels de sistema como '$image' e tokens relacionados à imagem.
        """
        try:
            cfg = getattr(project, "label_config", "") or ""
            labels = []
            # busca especificamente por tags <Label ... value="..."/>
            for m in re.finditer(r'<Label[^>]*\bvalue\s*=\s*["\']([^"\']+)["\']', cfg, flags=re.IGNORECASE):
                labels.append(m.group(1).strip())
            # fallback: procurar por nomes entre <Label>...</Label>
            for m in re.finditer(r'<Label[^>]*>([^<]+)</Label>', cfg, flags=re.IGNORECASE):
                labels.append(m.group(1).strip())
            # filtrar duplicatas e remover tokens de sistema (ex.: iniciando com '$' ou 'image')
            seen = set()
            out = []
            for l in labels:
                if not l:
                    continue
                low = l.strip()
                if low in seen:
                    continue
                # ignorar labels de sistema
                if low.startswith('$') or low.lower() == 'image' or 'image' in low.lower():
                    continue
                seen.add(low)
                out.append(low)
            return out
        except Exception as e:
            logger.error(f"Erro ao parsear label_config: {e}")
            return []

    def _sam_predict(self, img_url, task_id=None):
        """Faz a predição utilizando SAM2 AutomaticMaskGenerator quando disponível."""
        try:
            image = self._set_image(img_url, task_id)
            height, width = image.shape[:2]
            image_area = height * width

            # Tenta usar o Automatic Mask Generator do sam2
            amg = None
            amg_cls = None
            try:
                # tenta import comum
                from sam2.sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator as AMG
                amg_cls = AMG
            except Exception:
                try:
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as AMG
                    amg_cls = AMG
                except Exception:
                    amg_cls = None

            masks_list = []
            scores_list = []

            if amg_cls is not None:
                try:
                    # Instancia e gera máscaras automaticamente
                    amg = amg_cls(self.predictor.model)
                    amg_results = amg.generate(image)
                    # amg_results expected: list of dicts containing mask-like data (segmentation/mask) and score fields
                    for r in amg_results:
                        # vários formatos possíveis: 'segmentation', 'mask' or 'segmentation_mask'
                        seg = r.get("segmentation") or r.get("mask") or r.get("segmentation_mask") or r.get("pred_mask")
                        if seg is None:
                            continue
                        seg_arr = np.asarray(seg, dtype=np.uint8)
                        # normalizar para 0..1 float
                        seg_float = (seg_arr > 0).astype(np.float32)
                        masks_list.append(seg_float)
                        score = r.get("predicted_iou", r.get("score", r.get("stability_score", 1.0)))
                        scores_list.append(float(score))
                    logger.info(f"AMG gerou {len(masks_list)} máscaras")
                except Exception as e:
                    logger.error(f"AMG falhou: {e}")
                    amg_cls = None

            if amg_cls is None:
                # Se AMG não disponível, informar instrução clara para instalação
                raise RuntimeError(
                    "AutomaticMaskGenerator não encontrado no pacote sam2. "
                    "Instale/atualize o pacote sam2 que fornece SAM2AutomaticMaskGenerator "
                    "ou adicione um gerador automático. Ex.: pip install sam2[auto] ou verifique a versão."
                )

            if not masks_list:
                logger.warning("Nenhuma máscara gerada pelo AMG; abortando predição.")
                return {"masks": [], "probs": [], "width": width, "height": height, "class_labels": []}

            # Ordena por score
            inds = np.argsort(np.array(scores_list))[::-1]
            masks = [masks_list[i] for i in inds]
            scores = [scores_list[i] for i in inds]

            # Filtra e atribui classes por heurística de área (como antes)
            masks_normalized = [m.astype(np.float32) for m in masks]

            filtering_enabled = self.classes_config.get("filtering_rules", {}).get("enabled", False)
            min_area_pct = self.classes_config.get("filtering_rules", {}).get("min_mask_area_percentage", 0.05)
            max_area_pct = self.classes_config.get("filtering_rules", {}).get("max_mask_area_percentage", 95.0)

            filtered_masks = []
            filtered_scores = []
            filtered_labels = []

            for mask, score in zip(masks_normalized, scores):
                area_frac = float(mask.sum()) / float(image_area)
                mask_area_percentage = area_frac * 100.0

                if filtering_enabled and (mask_area_percentage < min_area_pct or mask_area_percentage > max_area_pct):
                    logger.debug(f"Máscara filtrada por tamanho: {mask_area_percentage:.4f}%")
                    continue

                label = self._assign_label_by_size(area_frac)
                logger.debug(f"Máscara com área {mask_area_percentage:.2f}% atribuída à classe '{label}'")

                filtered_masks.append(mask)
                filtered_scores.append(float(score))
                filtered_labels.append(label)

            if not filtered_masks:
                logger.warning("Nenhuma máscara passou pelos filtros. Retornando todas as máscaras geradas.")
                filtered_masks = masks_normalized
                filtered_scores = [float(s) for s in scores]
                filtered_labels = [self._assign_label_by_size((np.array(m).sum() / float(image_area))) for m in masks_normalized]

            logger.info(f"Predição gerada: {len(filtered_masks)} máscaras após filtragem")
            return {
                "masks": filtered_masks,
                "probs": filtered_scores,
                "width": width,
                "height": height,
                "class_labels": filtered_labels
            }

        except Exception as e:
            logger.error(f"Erro na predição SAM2: {e}")
            raise

    def _submit_predictions(self, task_id, predictions, project_id):
        """Submete as predições ao Label Studio usando Project.create_prediction"""
        if not self.client:
            logger.error("Cliente Label Studio não está disponível")
            return

        try:
            project = self.client.get_project(project_id)
            # Project.create_prediction expected signature: task_id, result, model_version, score
            project.create_prediction(
                task_id=task_id,
                result=predictions[0]["result"],
                model_version="sam2-auto-label",
                score=predictions[0]["score"]
            )
            logger.info(f"Predições submetidas para a tarefa {task_id}")

        except Exception as e:
            logger.error(f"Erro ao submeter predições para tarefa {task_id}: {e}")

    def _set_image(self, image_url, task_id=None):
        """Carrega a imagem (suporta URLs http(s), caminhos /data/... ou locais).
        Nota: não chama self.predictor.set_image para evitar dupla computação de embeddings
        quando for usado o AutomaticMaskGenerator.
        """
        try:
            ls_base = os.getenv('LABEL_STUDIO_URL', 'http://127.0.0.1:8080').rstrip('/')
            token = os.getenv('LABEL_STUDIO_API_KEY') or os.getenv('API_PERSONAL_ACESS_TOKEN') or os.getenv('API_KEY')
            def fetch_url(url, use_auth=False):
                headers = {'Authorization': f'Token {token}'} if (use_auth and token) else None
                resp = requests.get(url, timeout=30, headers=headers)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content))

            if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                use_auth = image_url.startswith(ls_base)
                image = fetch_url(image_url, use_auth=use_auth)
            else:
                if isinstance(image_url, str) and image_url.startswith('/'):
                    full_url = ls_base + image_url
                    try:
                        image = fetch_url(full_url, use_auth=True)
                    except Exception:
                        if os.path.exists(image_url):
                            image = Image.open(image_url)
                        else:
                            raise
                else:
                    if isinstance(image_url, str) and os.path.exists(image_url):
                        image = Image.open(image_url)
                    else:
                        full_url = ls_base + '/' + image_url.lstrip('/') if isinstance(image_url, str) else None
                        if full_url:
                            try:
                                image = fetch_url(full_url, use_auth=True)
                            except Exception:
                                raise FileNotFoundException(f"Cannot resolve image path: {image_url}")
                        else:
                            raise FileNotFoundException(f"Cannot resolve image path: {image_url}")

            image = np.array(image.convert("RGB"))
            # NÃO chamar self.predictor.set_image(image) aqui para evitar dupla computação de embeddings
            logger.info(f"Imagem carregada com sucesso (task={task_id}, shape={image.shape})")
            return image
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {image_url}: {e}")
            raise
