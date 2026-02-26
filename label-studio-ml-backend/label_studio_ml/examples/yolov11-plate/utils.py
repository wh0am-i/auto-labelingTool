import os


class Utils:

    @staticmethod
    def get_absolute_url(url):
        if not url:
            return url
        # Se a URL já começar com http ou https, não faz nada
        if url.startswith("http://") or url.startswith("https://"):
            return url

        # Pega a URL base do ambiente
        base_url = os.getenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080").rstrip("/")

        # Garante que o caminho comece com /
        if not url.startswith("/"):
            url = "/" + url

        return f"{base_url}{url}"

    def _resolve_task_media(self, data):
        data = data or {}
        raw = data.get("image") or data.get("image_url")
        media_type = "image"
        if not raw and (data.get("video") or data.get("video_url")):
            raw = data.get("video") or data.get("video_url")
            media_type = "video"
        if not raw and data:
            raw = list(data.values())[0]
        if isinstance(raw, str):
            lower = raw.lower().split("?", 1)[0]
            if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                media_type = "video"
        return media_type, self.get_absolute_url(raw)
