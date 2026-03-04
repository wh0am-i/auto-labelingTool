import os  # Manipulação de variáveis de ambiente e arquivos
import tempfile  # Criação de arquivos temporários
import requests  # Download do vídeo via HTTP
import numpy as np  # Cálculo de média de score
from uuid import uuid4  # Geração de IDs únicos
from PIL import Image  # Conversão de array para imagem PIL

# Tenta importar OpenCV (necessário para ler vídeo)
try:
    import cv2
except Exception:
    cv2 = None  # Se não estiver instalado, evita crash imediato

from image import ImageProcessor  # Classe responsável pela predição em imagem


class VideoProcessor:
    def __init__(self, image_processor):
        # Recebe uma instância do ImageProcessor
        # Isso permite reutilizar a lógica de predição de imagem para cada frame
        self.image = image_processor

    def _predict_video(self, video_url):
        # Garante que o OpenCV está disponível
        if cv2 is None:
            raise RuntimeError(
                "opencv-python nao esta instalado; necessario para auto-labeling de video."
            )

        # Obtém token do Label Studio para autenticação
        token = os.getenv("LEGACY_TOKEN") or os.getenv("PERSONAL_TOKEN")
        headers = {"Authorization": f"Token {token}"} if token else {}

        # Faz download do vídeo em modo streaming
        resp = requests.get(video_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()  # Lança erro se falhar

        # Define extensão do vídeo (ex: .mp4)
        suffix = os.path.splitext(video_url.split("?")[0])[1] or ".mp4"

        # Cria arquivo temporário para salvar o vídeo
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            temp_video_path = tf.name
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tf.write(chunk)

        frame_items = []  # Lista que armazenará todas as detecções
        frames_count = 0  # Total de frames do vídeo
        duration = 0.0  # Duração total do vídeo em segundos

        try:
            # Abre o vídeo com OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Falha ao abrir video: {video_url}")

            # FPS nominal do stream de vídeo
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps <= 0:
                src_fps = float(os.getenv("YOLO_VIDEO_SRC_FPS_FALLBACK", 30.0))

            # Total de frames informado pelo container/codec (pode ser 0 em alguns formatos)
            total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Define stride (de quantos em quantos frames rodar inferência)
            frame_stride = int(os.getenv("YOLO_VIDEO_FRAME_STRIDE", 1))
            if frame_stride < 1:
                frame_stride = 1

            # Base de indexação (Label Studio geralmente começa do frame 0)
            frame_base = int(os.getenv("YOLO_VIDEO_FRAME_BASE", 0))
            if frame_base not in (0, 1):
                frame_base = 0

            frame_idx = -1  # Índice do frame atual
            last_decoded_idx = -1  # Último frame efetivamente lido

            # Loop principal de leitura de frames
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # Sai se não houver mais frames

                frame_idx += 1
                last_decoded_idx = frame_idx

                # Aplica stride (pula frames se necessário)
                if frame_idx % frame_stride != 0:
                    continue

                # Converte frame BGR (OpenCV) para RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Converte array numpy para imagem PIL
                pil_img = Image.fromarray(rgb)

                # Faz predição no frame usando o ImageProcessor
                pred = self.image._predict_pil_image(pil_img)

                # Tempo determinístico por índice de frame.
                # CAP_PROP_POS_MSEC pode vir distorcido dependendo do backend/codec.
                time_sec = frame_idx / src_fps

                # Ajusta índice conforme base configurada
                ls_frame_idx = frame_idx + frame_base

                # Para cada detecção encontrada no frame
                for it in pred["items"]:
                    frame_items.append(
                        {
                            "id": str(uuid4()),  # ID único da detecção
                            "label": it.get("label", "object"),  # Classe detectada
                            "score": float(it.get("score", 0.0)),  # Confiança
                            "frame": ls_frame_idx,  # Frame da detecção
                            "frame_start": ls_frame_idx,
                            "frame_end": ls_frame_idx + frame_stride,
                            "time": float(time_sec),  # Tempo em segundos
                            "x": float(it.get("x", 0)),  # Posição X (%)
                            "y": float(it.get("y", 0)),  # Posição Y (%)
                            "width": float(it.get("width", 0)),  # Largura (%)
                            "height": float(it.get("height", 0)),  # Altura (%)
                            "rotation": float(it.get("rotation", 0)),  # Rotação
                            # enabled=False evita que o Label Studio estenda a caixa automaticamente
                            "enabled": False,
                        }
                    )

            # Usa quantidade real decodificada quando metadata for ausente/inconsistente.
            decoded_frames = max(0, last_decoded_idx + 1)
            if total_frames_meta > 0 and decoded_frames > 0:
                # Em alguns codecs/containers o CAP_PROP_FRAME_COUNT pode divergir;
                # usamos o menor para manter coerência com os frames realmente processáveis.
                frames_count = min(total_frames_meta, decoded_frames)
            elif total_frames_meta > 0:
                frames_count = total_frames_meta
            else:
                frames_count = max(1, decoded_frames)

            duration = float(frames_count / src_fps)

            cap.release()  # Libera vídeo

        finally:
            # Remove arquivo temporário
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        # Retorna estrutura intermediária
        return {
            "frame_items": frame_items,
            "frames_count": frames_count,
            "duration": duration,
        }

    def _generate_video_results(self, video_pred):
        # Obtém nomes configurados no Label Studio
        from_name = os.getenv("LS_VIDEO_FROM_NAME", "label")
        to_name = os.getenv("LS_VIDEO_TO_NAME", "video")

        results = []

        total_frames = max(1, int(video_pred.get("frames_count", 0)))
        duration = float(video_pred.get("duration", 0.0))

        # Converte cada detecção para formato esperado pelo Label Studio
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
                        "framesCount": total_frames,  # Total de frames do vídeo
                        "duration": duration,  # Duração total em segundos
                    },
                }
            )

        # Calcula score médio das detecções
        avg = float(np.mean([r.get("score", 0.0) for r in results])) if results else 0.0

        # Retorna no formato final que o Label Studio espera
        return [{"result": results, "model_version": "yolov11-video", "score": avg}]
