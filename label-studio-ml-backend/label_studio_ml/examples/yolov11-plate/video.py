import os
import tempfile
import requests
import numpy as np
from uuid import uuid4
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from image import ImageProcessor


class VideoProcessor:
    def __init__(self, image_processor):
        self.image = image_processor

    def _predict_video(self, video_url):
        if cv2 is None:
            raise RuntimeError(
                "opencv-python nao esta instalado; necessario para auto-labeling de video."
            )

        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        headers = {"Authorization": f"Token {token}"} if token else {}
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
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Falha ao abrir video: {video_url}")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if src_fps <= 0:
                src_fps = float(os.getenv("YOLO_VIDEO_SRC_FPS_FALLBACK", 30.0))
            total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_stride = int(os.getenv("YOLO_VIDEO_FRAME_STRIDE", 1))
            if frame_stride < 1:
                frame_stride = 1
            # Label Studio generally expects frame indexes starting at 0.
            frame_base = int(os.getenv("YOLO_VIDEO_FRAME_BASE", 0))
            if frame_base not in (0, 1):
                frame_base = 0

            frame_idx = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % frame_stride != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pred = self.image._predict_pil_image(pil_img)
                ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                time_sec = (
                    float(ts_msec / 1000.0) if ts_msec > 0 else (frame_idx / src_fps)
                )
                ls_frame_idx = frame_idx + frame_base
                duration = max(duration, time_sec)

                for it in pred["items"]:
                    frame_items.append(
                        {
                            "id": str(uuid4()),
                            "label": it.get("label", "object"),
                            "score": float(it.get("score", 0.0)),
                            "frame": ls_frame_idx,
                            "frame_start": ls_frame_idx,
                            "frame_end": ls_frame_idx,
                            "time": float(time_sec),
                            "x": float(it.get("x", 0)),
                            "y": float(it.get("y", 0)),
                            "width": float(it.get("width", 0)),
                            "height": float(it.get("height", 0)),
                            "rotation": float(it.get("rotation", 0)),
                            # Each detection is frame-local. If enabled=True without a closing
                            # keyframe, Label Studio can extend the box lifespan and visually
                            # "repeat" old boxes in following frames.
                            "enabled": False,
                        }
                    )

            # Keep full video metadata independent from stride, to avoid timeline drift.
            if total_frames_meta > 0:
                frames_count = total_frames_meta
                duration = max(duration, float(total_frames_meta / src_fps))
            else:
                frames_count = max(1, frame_idx + 1)
                duration = max(duration, float(frames_count / src_fps))
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
        }

    def _generate_video_results(self, video_pred):
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
