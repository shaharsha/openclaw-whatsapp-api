"""Agentleh WhatsApp Bridge — connects OpenClaw to WhatsApp via Meta Cloud API.

Run with: uvicorn bridge:app --port 8000
"""

import asyncio
import base64
import json as _json
import logging
import os
import re
from pathlib import Path

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

from meta_client import MetaWhatsAppClient
from gateway_client import OpenClawGateway

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("agentleh")

# --- Config ---

OPENCLAW_STATE_DIR = Path(
    os.environ.get("OPENCLAW_STATE_DIR", str(Path.home() / ".openclaw"))
)
WHATSAPP_ACCESS_TOKEN = os.environ["WHATSAPP_ACCESS_TOKEN"]
WHATSAPP_PHONE_NUMBER_ID = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
WEBHOOK_VERIFY_TOKEN = os.environ.get("WEBHOOK_VERIFY_TOKEN", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "ws://127.0.0.1:18789")
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")
SESSION_KEY = os.environ.get("SESSION_KEY", "main")
WABA_ID = os.environ["WHATSAPP_WABA_ID"]

# --- Globals ---

meta: MetaWhatsAppClient = None
gateway: OpenClawGateway = None
_message_lock: asyncio.Lock | None = None  # serializes message handling


_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_STRIKE = re.compile(r"~~(.+?)~~")
_RE_HEADING = re.compile(r"(?m)^#{1,3} +(.+)$")
_RE_BLOCKQUOTE = re.compile(r"(?m)^> ?")


def md_to_whatsapp(text: str) -> str:
    """Convert Markdown formatting to WhatsApp-compatible formatting.

    - **bold** → *bold*
    - *italic* → _italic_ (only single asterisks that aren't part of bold)
    - ~~strike~~ → ~strike~
    - ## heading → *heading*
    - > blockquote → removed
    """
    bold_marker = "\x01"
    result = _RE_BOLD.sub(bold_marker + r"\1" + bold_marker, text)
    result = _RE_ITALIC.sub(r"_\1_", result)
    result = result.replace(bold_marker, "*")
    result = _RE_STRIKE.sub(r"~\1~", result)
    result = _RE_HEADING.sub(r"*\1*", result)
    result = _RE_BLOCKQUOTE.sub("", result)
    return result


def normalize_phone(phone: str) -> str:
    """Strip non-digits from a phone number."""
    return re.sub(r"\D", "", phone)


def _read_openclaw_config() -> dict:
    """Read OpenClaw config from OPENCLAW_STATE_DIR (no caching — agent may modify it)."""
    config_path = OPENCLAW_STATE_DIR / "openclaw.json"
    try:
        with open(config_path) as f:
            return _json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _get_allowed_numbers() -> set[str]:
    """Read channels.whatsapp.allowFrom from OpenClaw config."""
    cfg = _read_openclaw_config()
    allow_from = cfg.get("channels", {}).get("whatsapp", {}).get("allowFrom", [])
    return {normalize_phone(n) for n in allow_from}


def derive_session_key(phone: str) -> str:
    """Derive session key based on OpenClaw's session.dmScope config.

    "main" = all users share one global session
    "per-peer" / "per-channel-peer" = each phone number gets isolated session
    """
    scope = _read_openclaw_config().get("session", {}).get("dmScope", "main")
    if scope in ("per-peer", "per-channel-peer", "per-account-channel-peer"):
        digits = normalize_phone(phone)
        return f"{SESSION_KEY}-wa-{digits}"
    return SESSION_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta, gateway, _message_lock

    _message_lock = asyncio.Lock()
    meta = MetaWhatsAppClient(WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID)
    gateway = OpenClawGateway(
        url=GATEWAY_URL,
        token=GATEWAY_TOKEN,
        session_key=SESSION_KEY,
        sessions_dir=str(OPENCLAW_STATE_DIR / "agents" / "main" / "sessions"),
    )

    try:
        await gateway.connect()
        logger.info("Bridge started — gateway connected, webhook ready")
    except Exception as e:
        logger.error("Failed to connect to OpenClaw gateway: %s", e)
        logger.info("Bridge started in webhook-only mode (gateway offline)")

    yield

    await gateway.close()
    await meta.close()
    logger.info("Bridge stopped")


app = FastAPI(title="Agentleh WhatsApp Bridge", lifespan=lifespan)


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification challenge."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
        logger.info("Webhook verified")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Webhook verification failed (token mismatch)")
    return Response(status_code=403)


def _extract_message(msg: dict, msg_type: str) -> tuple[str, dict | None]:
    """Extract text and media info from any WhatsApp message type.

    Returns (text, media_info) where media_info is None for text-only messages,
    or a dict with media_id, mime_type, type, and optional filename/caption.
    """
    if msg_type == "text":
        return msg.get("text", {}).get("body", ""), None

    elif msg_type == "image":
        info = msg.get("image", {})
        caption = info.get("caption", "")
        return caption or "User sent an image.", {
            "media_id": info.get("id", ""),
            "mime_type": info.get("mime_type", "image/jpeg"),
            "type": "image",
        }

    elif msg_type == "audio":
        info = msg.get("audio", {})
        return "User sent a voice message.", {
            "media_id": info.get("id", ""),
            "mime_type": info.get("mime_type", "audio/ogg"),
            "type": "audio",
        }

    elif msg_type == "video":
        info = msg.get("video", {})
        caption = info.get("caption", "")
        return caption or "User sent a video.", {
            "media_id": info.get("id", ""),
            "mime_type": info.get("mime_type", "video/mp4"),
            "type": "video",
        }

    elif msg_type == "document":
        info = msg.get("document", {})
        filename = info.get("filename", "document")
        caption = info.get("caption", "")
        return caption or f"User sent a document: {filename}", {
            "media_id": info.get("id", ""),
            "mime_type": info.get("mime_type", "application/octet-stream"),
            "type": "document",
            "filename": filename,
        }

    elif msg_type == "sticker":
        info = msg.get("sticker", {})
        return "User sent a sticker.", {
            "media_id": info.get("id", ""),
            "mime_type": info.get("mime_type", "image/webp"),
            "type": "sticker",
        }

    elif msg_type == "location":
        loc = msg.get("location", {})
        name = loc.get("name", "")
        addr = loc.get("address", "")
        lat, lon = loc.get("latitude", ""), loc.get("longitude", "")
        return f"[location] {name} {addr} ({lat}, {lon})".strip(), None

    elif msg_type == "contacts":
        contacts = msg.get("contacts", [])
        names = [c.get("name", {}).get("formatted_name", "") for c in contacts]
        return f"[contacts] {', '.join(names)}", None

    elif msg_type == "reaction":
        emoji = msg.get("reaction", {}).get("emoji", "")
        return f"[reaction] {emoji}", None

    else:
        return f"[{msg_type} message]", None


@app.post("/webhook")
async def receive_webhook(request: Request):
    """Receive incoming WhatsApp messages from Meta."""
    body = await request.json()

    # Meta sends the webhook in this format:
    # { "object": "whatsapp_business_account", "entry": [...] }
    if body.get("object") != "whatsapp_business_account":
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            contacts = value.get("contacts", [])

            for msg in messages:
                msg_type = msg.get("type", "")
                sender = msg.get("from", "")
                msg_id = msg.get("id", "")
                text, media_info = _extract_message(msg, msg_type)
                sender_name = ""

                # Get sender name from contacts
                for contact in contacts:
                    if contact.get("wa_id") == sender:
                        profile = contact.get("profile", {})
                        sender_name = profile.get("name", sender)
                        break

                if not text and not media_info:
                    continue

                # Check allowlist from openclaw.json channels.whatsapp.allowFrom
                allowed = _get_allowed_numbers()
                if allowed and normalize_phone(sender) not in allowed:
                    logger.warning("Blocked message from non-allowed number: %s", sender)
                    continue

                logger.info(
                    "Incoming message from %s (%s): %s%s",
                    sender,
                    sender_name,
                    text[:100],
                    f" [+media: {media_info['type']}]" if media_info else "",
                )

                # Process in background so we return 200 immediately
                asyncio.create_task(
                    handle_message(sender, sender_name, text, msg_id, media_info)
                )

    return {"status": "ok"}


async def handle_message(
    sender: str,
    sender_name: str,
    text: str,
    msg_id: str,
    media_info: dict | None = None,
):
    """Forward a message to OpenClaw and relay the reply back to WhatsApp.

    Serialized via _message_lock to prevent concurrent JSONL polls from
    stealing each other's replies (see test_concurrent_tasks_steal_each_others_replies).
    """
    async with _message_lock:
        try:
            session_key = derive_session_key(sender)
            logger.info("Routing to session: %s", session_key)

            # Download media if present
            attachments = None
            if media_info and media_info.get("media_id"):
                try:
                    data, mime = await meta.download_media(media_info["media_id"])
                    ext = mime.split("/")[-1].split(";")[0]
                    filename = media_info.get("filename", f"media.{ext}")
                    media_type = media_info.get("type", "")

                    if media_type in ("image", "sticker"):
                        attachments = [{
                            "mimeType": mime,
                            "fileName": filename,
                            "content": base64.b64encode(data).decode(),
                        }]
                    else:
                        workspace = OPENCLAW_STATE_DIR / "workspace"
                        media_dir = workspace / "media"
                        media_dir.mkdir(parents=True, exist_ok=True)
                        save_path = media_dir / filename
                        save_path.write_bytes(data)
                        text += f"\n[File saved to workspace: media/{filename}]"

                    logger.info(
                        "Downloaded media: %s (%s, %d bytes, action=%s)",
                        filename, mime, len(data),
                        "attach" if attachments else "save",
                    )
                except Exception as e:
                    logger.error("Failed to download media: %s", e)
                    text += f"\n(Media download failed: {e})"

            msg_count = 0
            read_marked = False

            async def on_agent_message(reply_text: str):
                nonlocal msg_count, read_marked
                # Mark as read on first reply (blue checks appear when agent responds)
                if not read_marked:
                    await meta.mark_read(msg_id)
                    read_marked = True
                reply_text = md_to_whatsapp(reply_text)
                chunks = split_message(reply_text, max_len=4000)
                for chunk in chunks:
                    await meta.send_text(sender, chunk)
                    if len(chunks) > 1:
                        await asyncio.sleep(0.3)
                msg_count += 1
                logger.info("Sent message %d to %s (%d chars)", msg_count, sender, len(reply_text))

            await gateway.send_and_stream(
                text=text,
                on_message=on_agent_message,
                sender=f"+{sender}",
                sender_name=sender_name,
                role="admin",
                session_key=session_key,
                attachments=attachments,
            )

            # Mark read even if agent returned empty (so message doesn't stay unread)
            if not read_marked:
                await meta.mark_read(msg_id)

            if msg_count == 0:
                logger.warning("No replies from agent for session %s", session_key)

        except TimeoutError:
            logger.error("Timeout waiting for agent reply (sender=%s)", sender)
            await meta.send_text(sender, "הסוכן לא הגיב בזמן. נסה שוב.")
        except Exception as e:
            logger.error("Error handling message from %s: %s", sender, e, exc_info=True)
            await meta.send_text(sender, "משהו השתבש. נסה שוב.")


def split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split a long message into WhatsApp-friendly chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at paragraph break
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at < max_len // 4:
            # Try single newline
            split_at = text.rfind("\n", 0, max_len)
        if split_at < max_len // 4:
            # Try sentence end
            for sep in [". ", "? ", "! "]:
                split_at = text.rfind(sep, 0, max_len)
                if split_at >= max_len // 4:
                    split_at += 1  # include the punctuation
                    break
        if split_at < max_len // 4:
            # Hard split at space
            split_at = text.rfind(" ", 0, max_len)
        if split_at < max_len // 4:
            # Last resort: hard split
            split_at = max_len

        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    return chunks


@app.get("/api/templates")
async def api_list_templates():
    """List all approved WhatsApp message templates with their parameters."""
    raw = await meta.list_templates(WABA_ID)
    templates = []
    for t in raw:
        body_text = ""
        param_count = 0
        for comp in t.get("components", []):
            if comp.get("type") == "BODY":
                body_text = comp.get("text", "")
                param_count = len(re.findall(r"\{\{\d+\}\}", body_text))
        templates.append({
            "name": t["name"],
            "language": t.get("language", ""),
            "category": t.get("category", ""),
            "body": body_text,
            "parameter_count": param_count,
        })
    return {"templates": templates}


@app.post("/api/send")
async def api_send_message(request: Request):
    """Send a WhatsApp message to any number.

    Body JSON:
      - to: phone number (international, no +)
      - text: message text (for free-form within 24hr window)
      - template_name: template to use (optional, defaults to hello_greeting)
      - template_params: list of strings for template variables {{1}}, {{2}}...
      - language: template language code (optional, defaults to en_US)
      - force_template: if true, skip free-form and send template directly
    """
    body = await request.json()
    to = normalize_phone(body.get("to", ""))
    text = body.get("text", "")
    template_name = body.get("template_name", "")
    template_params = body.get("template_params", None)
    language = body.get("language", "en_US")
    force_template = body.get("force_template", False)

    if not to:
        return {"error": "Missing 'to' field"}

    if force_template or (not text):
        if not template_name:
            return {
                "error": "Missing 'template_name'. Use GET /api/templates to see available templates.",
            }
        # Send template directly
        result = await meta.send_template(to, template_name, language, template_params)
        if result.get("error"):
            return {"status": "error", "error": result["error"]}
        return {
            "status": "sent_template",
            "template": template_name,
            "parameters": template_params,
            "result": result,
        }

    # Try free-form first
    result = await meta.send_text(to, text)

    # Check if outside 24hr window (error code 131047 or message about re-engagement)
    err = result.get("error", {})
    if err.get("code") in (131047, 131026) or "re-engagement" in str(err).lower():
        logger.info(
            "Outside 24hr window for %s — cannot send free-form text. "
            "Use force_template=true with a template instead.",
            to,
        )
        return {
            "status": "failed_outside_window",
            "error": "Cannot send free-form message: outside 24-hour conversation window. "
                     "The recipient hasn't messaged this number recently. "
                     "You must use a template message instead. "
                     "Call this endpoint again with force_template=true and a template_name. "
                     "Use GET /api/templates to see available templates and their parameters.",
            "available_action": {
                "method": "POST",
                "url": "http://localhost:8000/api/send",
                "body": {
                    "to": to,
                    "force_template": True,
                    "template_name": "<pick from /api/templates>",
                    "template_params": ["<param1>", "<param2>"],
                },
            },
        }

    if err:
        return {"status": "error", "error": err}

    return {"status": "sent_freeform", "result": result}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gateway_connected": gateway is not None and gateway._ws is not None and gateway._ws.close_code is None,
    }
