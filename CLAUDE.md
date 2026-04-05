# openclaw-whatsapp-api

Generic WhatsApp bridge for OpenClaw. Connects any OpenClaw agent to WhatsApp via Meta's official Cloud API.

## Architecture

```
WhatsApp User ←→ Meta Cloud API ←→ Bridge (FastAPI :8000) ←→ OpenClaw Gateway (WS) ←→ LLM
```

- **bridge.py** — FastAPI webhook server. Receives WhatsApp messages, routes to OpenClaw, relays agent replies back. Handles media, markdown-to-WhatsApp formatting, message splitting, allowlist, session routing.
- **gateway_client.py** — OpenClaw WebSocket client. Ed25519 device signing, challenge-response auth, JSONL polling for agent replies.
- **meta_client.py** — Meta WhatsApp Cloud API client. Send text/templates, download media, mark read.

## Running

```bash
uv run uvicorn bridge:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables (.env)

```
WHATSAPP_ACCESS_TOKEN=       # Meta System User token
WHATSAPP_PHONE_NUMBER_ID=    # Meta phone number ID
WHATSAPP_WABA_ID=            # WhatsApp Business Account ID
WEBHOOK_VERIFY_TOKEN=        # Webhook verification token
GATEWAY_URL=ws://127.0.0.1:18789
GATEWAY_TOKEN=               # OpenClaw gateway auth token
SESSION_KEY=main
OPENCLAW_STATE_DIR=          # Path to OpenClaw state directory
```

## Key Design Decisions

- **JSONL polling over WebSocket events**: WS events miss post-tool-call text. JSONL is source of truth.
- **asyncio.Lock**: Serializes message handling to prevent concurrent JSONL poll stealing.
- **Mark-read on reply**: Blue checks appear when agent responds, not on receipt.
- **Session scope from config**: Reads `session.dmScope` from openclaw.json each time.
- **Allowlist from config**: Reads `channels.whatsapp.allowFrom` fresh each time (agent can modify).
- **Media routing**: Images/stickers → base64 attachments (vision). Audio/video/docs → workspace/media/.

## Testing

```bash
uv run pytest
```

97 tests covering JSONL polling, settle timer, media handling, concurrent access, gateway reconnection.
