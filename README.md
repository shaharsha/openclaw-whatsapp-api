# openclaw-whatsapp-api

A WhatsApp bridge for [OpenClaw](https://openclaw.ai) using Meta's official Cloud API.

Connects any OpenClaw agent to WhatsApp with zero ban risk — no Baileys, no unofficial libraries.

## Features

- Full media support (images via vision, audio/video/docs saved to workspace)
- Tool calling and multi-step agent workflows
- Template message support (for initiating conversations outside 24hr window)
- Per-number allowlist (configurable by the agent at runtime)
- Session isolation (shared or per-user)
- Auto-reconnection with exponential backoff
- Markdown → WhatsApp formatting conversion
- 97 tests

## How It Works

```
WhatsApp ←→ Meta Cloud API ←→ Bridge (FastAPI) ←→ OpenClaw Gateway (WebSocket) ←→ LLM
```

1. User sends a WhatsApp message
2. Meta delivers it to the bridge via webhook
3. Bridge forwards it to the OpenClaw agent via WebSocket (Ed25519 device signing)
4. Agent processes the message (may call tools, search the web, etc.)
5. Bridge polls the agent's session JSONL for replies
6. Replies are converted from Markdown to WhatsApp formatting and sent back

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [OpenClaw](https://openclaw.ai) running locally
- Meta WhatsApp Business account with Cloud API access
- A public URL for the webhook (e.g., [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/), ngrok)

### Setup

```bash
git clone https://github.com/shaharsha/openclaw-whatsapp-api.git
cd openclaw-whatsapp-api
uv sync
```

Create a `.env` file:

```env
WHATSAPP_ACCESS_TOKEN=       # Meta System User token
WHATSAPP_PHONE_NUMBER_ID=    # Meta phone number ID
WHATSAPP_WABA_ID=            # WhatsApp Business Account ID
WEBHOOK_VERIFY_TOKEN=        # Any string for webhook verification
GATEWAY_URL=ws://127.0.0.1:18789
GATEWAY_TOKEN=               # From your openclaw.json gateway.auth.token
SESSION_KEY=main
OPENCLAW_STATE_DIR=          # Path to your OpenClaw state directory
```

Run the bridge:

```bash
uv run uvicorn bridge:app --host 0.0.0.0 --port 8000 --reload
```

Set your Meta webhook URL to `https://<your-public-url>/webhook` with your verify token.

## Configuration

The bridge reads configuration from the OpenClaw state directory (`OPENCLAW_STATE_DIR`):

- **Session scope**: `session.dmScope` in `openclaw.json`
  - `main` — all users share one conversation
  - `per-channel-peer` — each phone number gets an isolated session
- **Allowlist**: `channels.whatsapp.allowFrom` in `openclaw.json` — array of phone numbers (e.g., `["+972546901044"]`). Empty = allow all.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook` | GET | Meta webhook verification |
| `/webhook` | POST | Receive incoming WhatsApp messages |
| `/api/send` | POST | Send a message (free-form or template) |
| `/api/templates` | GET | List approved message templates |
| `/health` | GET | Health check + gateway connection status |

## Testing

```bash
uv run pytest  # 97 tests, ~0.2s
```

## License

MIT
