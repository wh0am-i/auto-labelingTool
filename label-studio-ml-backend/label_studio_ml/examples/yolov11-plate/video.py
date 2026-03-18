import os
import sys
import tempfile
import subprocess
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

    def _render_progress(self, current, total, frame_idx, time_sec, src_fps):
        if total <= 0:
            return
        width = 40
        done = int(width * current / total)
        percent = (current / total) * 100
        bar = "[{}{}] {:5.1f}%".format("=" * done, " " * (width - done), percent)
        msg = f"{bar} frame:{frame_idx} time:{time_sec:.2f}s fps:{src_fps:.2f}"
        sys.stdout.write("\r" + msg + "   ")
        sys.stdout.flush()

    def _convert_to_cfr(self, input_path, target_fps):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        ffmpeg_path = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            input_path,
            "-vf",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

    def _process_video_frames(self, video_path, video_url):
        cap = cv2.VideoCapture(video_path)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 5.0)
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_items = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pred = self.image._predict_pil_image(pil_img)
            time_sec = frame_idx / src_fps
            self._render_progress(
                frame_idx,
                total_frames_meta,
                frame_idx,
                time_sec,
                src_fps,
            )
            for it in pred["items"]:
                it["frame"] = frame_idx
                it["time"] = time_sec
                frame_items.append(it)
        if total_frames_meta > 0:
            self._render_progress(
                total_frames_meta,
                total_frames_meta,
                frame_idx,
                frame_idx / src_fps,
                src_fps,
            )
            sys.stdout.write("\n")
            sys.stdout.flush()
        cap.release()
        return {
            "frame_items": frame_items,
            "frames_count": frame_idx,
            "duration": frame_idx / src_fps,
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
