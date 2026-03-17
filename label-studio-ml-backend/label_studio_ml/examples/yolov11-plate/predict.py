import os
import logging
import requests
import tempfile
from io import BytesIO
from PIL import Image
from uuid import uuid4

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, image_processor, video_processor, utils):
        self.model = model
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.utils = utils

    def _get_headers(self):
        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        return {"Authorization": f"Token {token}"} if token else {}

    def _download_to_file(self, url):
        resp = requests.get(url, headers=self._get_headers(), timeout=120, stream=True)
        resp.raise_for_status()
        suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                tf.write(chunk)
        tf.close()
        return tf.name

    def _download_to_memory(self, url):
        resp = requests.get(url, headers=self._get_headers(), timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def _get_predictions_for_task(self, task):
        data = task.get("data") or {}
        media_type, media_url = self.utils._resolve_task_media(data)

        if media_type == "video":
            # CORREÇÃO: Usa o método interno do Predictor para baixar
            temp_path = self._download_to_file(media_url)
            normalized_path = None
            converted_to_cfr = False

            import cv2

            cap = cv2.VideoCapture(temp_path)
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 5.0)
            target_fps = float(os.getenv("YOLO_VIDEO_TARGET_FPS", 5.0))
            cap.release()

            inference_path = temp_path
            if abs(src_fps - target_fps) > 0.01:
                normalized_path = self.video_processor._convert_to_cfr(
                    temp_path, target_fps
                )
                inference_path = normalized_path
                converted_to_cfr = True

            debug_path = self.video_processor._save_debug_video(
                inference_path, converted_to_cfr
            )

            # Aqui passamos o inference_path (que pode ser o original ou o convertido)
            video_pred = self.video_processor._process_video_frames(
                inference_path, media_url
            )

            video_pred.update(
                {
                    "converted_to_cfr": converted_to_cfr,
                    "processed_video_path": normalized_path,
                    "debug_video_path": debug_path,
                }
            )

            predictions = self.video_processor._generate_video_results(video_pred)
            return predictions, "video", inference_path

        else:
            img = self.image_processor.load_image_from_url(media_url)
            pred_data = self.image_processor._predict_pil_image(img)
            predictions = self.image_processor._generate_results(pred_data["items"])
            return predictions, "image", None
