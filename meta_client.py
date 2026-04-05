"""Meta WhatsApp Cloud API client."""

import httpx
import logging

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.facebook.com/v25.0"


class MetaWhatsAppClient:
    def __init__(self, access_token: str, phone_number_id: str):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.base_url = f"{GRAPH_API_BASE}/{phone_number_id}"
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def send_text(self, to: str, text: str) -> dict:
        """Send a free-form text message (only works within 24hr window)."""
        resp = await self._client.post(
            f"{self.base_url}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            logger.error("Failed to send text: %s", data)
        else:
            logger.info("Sent text to %s (status=%d)", to, resp.status_code)
        return data

    async def send_template(
        self,
        to: str,
        template_name: str,
        language: str = "en_US",
        parameters: list[str] | None = None,
    ) -> dict:
        """Send a template message (works outside 24hr window).

        Args:
            parameters: List of strings to fill template variables {{1}}, {{2}}, etc.
        """
        template_obj: dict = {
            "name": template_name,
            "language": {"code": language},
        }

        if parameters:
            template_obj["components"] = [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": p} for p in parameters
                    ],
                }
            ]

        resp = await self._client.post(
            f"{self.base_url}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "template",
                "template": template_obj,
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            logger.error("Failed to send template: %s", data)
        return data

    async def list_templates(self, waba_id: str) -> list[dict]:
        """Fetch all approved message templates from Meta."""
        templates = []
        url = f"{GRAPH_API_BASE}/{waba_id}/message_templates"
        params = {"fields": "name,status,language,components,category", "limit": 100}

        while url:
            resp = await self._client.get(url, params=params)
            data = resp.json()
            for t in data.get("data", []):
                if t.get("status") == "APPROVED":
                    templates.append(t)
            # Pagination
            url = data.get("paging", {}).get("next")
            params = None  # next URL includes params

        return templates

    async def download_media(
        self, media_id: str, max_bytes: int = 10_000_000
    ) -> tuple[bytes, str]:
        """Download media from WhatsApp.

        Two-step flow:
        1. GET /{media_id} → JSON with download URL
        2. GET {url} → binary data

        Returns (data, mime_type).
        """
        # Step 1: Get media info (URL + mime type)
        resp = await self._client.get(f"{GRAPH_API_BASE}/{media_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get media info: {resp.json()}")
        info = resp.json()
        url = info.get("url")
        mime = info.get("mime_type", "application/octet-stream")
        if not url:
            raise RuntimeError(f"No URL in media info: {info}")

        # Step 2: Download binary
        resp = await self._client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download media: status {resp.status_code}")
        data = resp.content
        if len(data) > max_bytes:
            raise ValueError(
                f"Media too large: {len(data)} bytes (max {max_bytes})"
            )

        logger.info("Downloaded media %s (%s, %d bytes)", media_id, mime, len(data))
        return data, mime

    async def mark_read(self, message_id: str) -> None:
        """Mark a message as read (blue checkmarks)."""
        await self._client.post(
            f"{self.base_url}/messages",
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            },
        )

    async def close(self):
        await self._client.aclose()
