import os
import tempfile
import shutil
import subprocess
import requests
import numpy as np
from uuid import uuid4
from PIL import Image

# Importa OpenCV se disponível
try:
    import cv2
except Exception:
    cv2 = None

from image import ImageProcessor


class VideoProcessor:
    def __init__(self, image_processor):
        # Usa ImageProcessor para inferência por frame
        self.image = image_processor

    def _convert_to_cfr(self, input_path, target_fps):
        # Converte vídeo para FPS constante (CFR)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        ffmpeg_path = shutil.which("ffmpeg")

        if not ffmpeg_path:
            raise RuntimeError("ffmpeg não encontrado no PATH.")

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-vf",
            f"fps={target_fps}",
            "-r",
            str(target_fps),
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            "-an",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-x264-params",
            "bframes=0",
            output_path,
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

    def _save_debug_video(self, source_video_path, converted_to_cfr):
        # Salva cópia do vídeo usado na inferência
        base_dir = os.path.dirname(__file__)
        debug_dir = os.path.join(base_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        prefix = "predicted_cfr" if converted_to_cfr else "predicted_src"
        debug_filename = f"{prefix}_{uuid4().hex}.mp4"
        debug_path = os.path.join(debug_dir, debug_filename)

        shutil.copy2(source_video_path, debug_path)
        return debug_path

    def _predict_video(self, video_url):
        # Garante OpenCV instalado
        if cv2 is None:
            raise RuntimeError(
                "opencv-python nao esta instalado; necessario para auto-labeling de video."
            )

        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        headers = {"Authorization": f"Token {token}"} if token else {}

        # Download do vídeo
        resp = requests.get(video_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()

        suffix = os.path.splitext(video_url.split("?")[0])[1] or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_video_path = tf.name
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tf.write(chunk)

        frame_items = []
        frames_count = 0
        duration = 0.0

        normalized_video_path = None
        video_path_for_infer = temp_video_path
        converted_to_cfr = False
        source_fps = 0.0
        effective_fps = 0.0
        debug_video_path = None

        try:
            cap = cv2.VideoCapture(video_path_for_infer)
            if not cap.isOpened():
                raise RuntimeError(f"Falha ao abrir video: {video_url}")

            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            source_fps = src_fps

            target_fps = float(os.getenv("YOLO_VIDEO_TARGET_FPS", 5.0))
            fps_tolerance = float(os.getenv("YOLO_VIDEO_TARGET_FPS_TOLERANCE", 0.00001))

            if src_fps <= 0:
                src_fps = float(os.getenv("YOLO_VIDEO_SRC_FPS_FALLBACK", target_fps))
                source_fps = src_fps

            # Converte para CFR se necessário
            if abs(src_fps - target_fps) > fps_tolerance:
                cap.release()
                normalized_video_path = self._convert_to_cfr(
                    temp_video_path, target_fps
                )
                video_path_for_infer = normalized_video_path
                converted_to_cfr = True

                cap = cv2.VideoCapture(video_path_for_infer)
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Falha ao abrir vídeo convertido: {video_path_for_infer}"
                    )

                src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if src_fps <= 0:
                    src_fps = target_fps

            effective_fps = src_fps

            # Salva vídeo processado
            debug_video_path = self._save_debug_video(
                video_path_for_infer, converted_to_cfr
            )

            total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            frame_stride = int(os.getenv("YOLO_VIDEO_FRAME_STRIDE", 1))
            if frame_stride < 1:
                frame_stride = 1

            frame_base = int(os.getenv("YOLO_VIDEO_FRAME_BASE", 0))
            if frame_base not in (0, 1):
                frame_base = 0

            frame_idx = 0
            last_decoded_idx = 0

            # Loop de leitura de frames
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                last_decoded_idx = frame_idx

                if frame_idx % frame_stride != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                pred = self.image._predict_pil_image(pil_img)

                time_sec = frame_idx / src_fps
                ls_frame_idx = frame_idx + frame_base

                print(
                    f"Frame idx: {frame_idx}; time_sec: {time_sec}; src_fps: {src_fps}"
                )

                for it in pred["items"]:
                    frame_items.append(
                        {
                            "id": str(uuid4()),
                            "label": it.get("label", "object"),
                            "score": float(it.get("score", 0.0)),
                            "frame": ls_frame_idx,
                            "frame_start": ls_frame_idx,
                            "frame_end": ls_frame_idx + frame_stride,
                            "time": float(time_sec),
                            "x": float(it.get("x", 0)),
                            "y": float(it.get("y", 0)),
                            "width": float(it.get("width", 0)),
                            "height": float(it.get("height", 0)),
                            "rotation": float(it.get("rotation", 0)),
                            "enabled": False,
                        }
                    )

            decoded_frames = max(0, last_decoded_idx + 1)

            if total_frames_meta > 0 and decoded_frames > 0:
                frames_count = min(total_frames_meta, decoded_frames)
            elif total_frames_meta > 0:
                frames_count = total_frames_meta
            else:
                frames_count = max(1, decoded_frames)

            duration = float(frames_count / src_fps)

            cap.release()

        finally:
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        return {
            "frame_items": frame_items,
            "frames_count": frames_count,
            "duration": duration,
            "source_fps": source_fps,
            "effective_fps": effective_fps,
            "converted_to_cfr": converted_to_cfr,
            "processed_video_path": normalized_video_path,
            "debug_video_path": debug_video_path,
        }

    def _generate_video_results(self, video_pred):
        # Converte predição para formato Label Studio
        from_name = os.getenv("LS_VIDEO_FROM_NAME", "label")
        to_name = os.getenv("LS_VIDEO_TO_NAME", "video")

        results = []

        total_frames = max(1, int(video_pred.get("frames_count", 0)))
        duration = float(video_pred.get("duration", 0.0))

        for item in video_pred.get("frame_items") or []:
            results.append(
                {
                    "id": str(uuid4()),
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "videorectangle",
                    "score": float(item.get("score", 0.0)),
                    "value": {
                        "labels": [item.get("label", "object")],
                        "sequence": [
                            {
                                "frame": int(item.get("frame", 0)),
                                "time": float(item.get("time", 0.0)),
                                "x": float(item.get("x", 0.0)),
                                "y": float(item.get("y", 0.0)),
                                "width": float(item.get("width", 0.0)),
                                "height": float(item.get("height", 0.0)),
                                "rotation": float(item.get("rotation", 0.0)),
                                "enabled": bool(item.get("enabled", True)),
                            }
                        ],
                        "framesCount": total_frames,
                        "duration": duration,
                    },
                }
            )

        avg = float(np.mean([r.get("score", 0.0) for r in results])) if results else 0.0

        return [{"result": results, "model_version": "yolov11-video", "score": avg}]
