# openclaw-whatsapp-api

A WhatsApp bridge for [OpenClaw](https://openclaw.ai) using Meta's official Cloud API.

Connects any OpenClaw agent to WhatsApp with zero ban risk — no Baileys, no unofficial libraries.

## Features

- Full media support (images via vision, audio/video/docs saved to workspace)
- Tool calling and multi-step agent workflows
- Template fallback outside 24hr window
- Per-number allowlist (configurable by the agent)
- Session isolation (shared or per-user)
- Auto-reconnection with exponential backoff
- Markdown → WhatsApp formatting conversion
- 97 tests

## Quick Start

```bash
git clone https://github.com/shaharsha/openclaw-whatsapp-api.git
cd openclaw-whatsapp-api
uv sync
cp .env.example .env  # fill in credentials
uv run uvicorn bridge:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

The bridge reads configuration from the OpenClaw state directory (`OPENCLAW_STATE_DIR`):

- **Session scope**: `session.dmScope` in `openclaw.json` (`main` = shared, `per-channel-peer` = per-user)
- **Allowlist**: `channels.whatsapp.allowFrom` in `openclaw.json`

## License

Private.
