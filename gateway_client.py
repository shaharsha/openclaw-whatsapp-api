"""OpenClaw Gateway WebSocket client.

Implements the OpenClaw protocol v3:
1. Connect via WebSocket with Ed25519 device signing
2. Send chat.send messages
3. Poll session JSONL for assistant replies (same approach as the Go kapso-whatsapp bridge)

The JSONL polling approach is used because OpenClaw's WebSocket chat events
don't reliably include post-tool-call text — that goes through the internal
reply dispatcher which only built-in channels (Baileys/Telegram) receive.
JSONL is the source of truth.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import websockets

logger = logging.getLogger(__name__)

DEVICE_KEY_PATH = Path.home() / ".config" / "agentleh" / "device-key.pem"

# JSONL polling config
POLL_INTERVAL = 1.5   # seconds between polls
POLL_TIMEOUT = 300.0  # max seconds to wait for a reply
SETTLE_TIME = 15.0    # seconds of no new activity before considering agent done
                      # must be long enough for tool calls (web search can take 10s+)


# ── Device identity helpers ──────────────────────────────────────────


def _load_or_create_device_key() -> Ed25519PrivateKey:
    DEVICE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEVICE_KEY_PATH.exists():
        pem = DEVICE_KEY_PATH.read_bytes()
        return serialization.load_pem_private_key(pem, password=None)
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    DEVICE_KEY_PATH.write_bytes(pem)
    logger.info("Generated new device key at %s", DEVICE_KEY_PATH)
    return key


def _device_id(key: Ed25519PrivateKey) -> str:
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return hashlib.sha256(pub).hexdigest()


def _public_key_b64(key: Ed25519PrivateKey) -> str:
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.urlsafe_b64encode(pub).rstrip(b"=").decode()


def _sign_device_payload(
    key: Ed25519PrivateKey, device_id: str, client_id: str, client_mode: str,
    role: str, token: str, scopes: list[str], signed_at_ms: int, nonce: str,
    plat: str,
) -> str:
    payload = "|".join([
        "v3", device_id, client_id, client_mode, role,
        ",".join(scopes), str(signed_at_ms), token, nonce,
        plat.lower().strip(), "",
    ])
    sig = key.sign(payload.encode())
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


# ── JSONL helpers ────────────────────────────────────────────────────


def _parse_ts(ts_str: str) -> float:
    """Parse an ISO timestamp string to epoch seconds."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def _find_session_file(sessions_json: str, session_key: str, base_key: str) -> str | None:
    """Read sessions.json and return the JSONL path for a session key."""
    try:
        with open(sessions_json) as f:
            sessions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    # Try canonical key first
    canonical = f"agent:{session_key}:{session_key}"
    if canonical in sessions:
        sf = sessions[canonical].get("sessionFile")
        if sf:
            return sf

    # Fallback: first entry containing the key
    for k, v in sessions.items():
        if session_key in k:
            sf = v.get("sessionFile")
            if sf:
                return sf

    # Try base key
    if session_key != base_key:
        return _find_session_file(sessions_json, base_key, base_key)

    return None


def _scan_jsonl(session_file: str, since: float) -> list[dict]:
    """Scan a session JSONL for all entries after `since`."""
    try:
        with open(session_file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    entries = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts_str = entry.get("timestamp", "")
        if ts_str:
            try:
                if _parse_ts(ts_str) < since:
                    continue
            except (ValueError, TypeError):
                continue

        entry["_line"] = i
        entries.append(entry)

    return entries


# ── Gateway client ───────────────────────────────────────────────────


class OpenClawGateway:
    def __init__(
        self,
        url: str = "ws://127.0.0.1:18789",
        token: str = "",
        session_key: str = "main",
        sessions_dir: str = "",
    ):
        self.url = url
        self.token = token
        self.session_key = session_key
        self.sessions_dir = sessions_dir
        self._ws = None
        self._seq = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._read_task = None
        self._claimed: set[str] = set()  # dedup: "filepath:line" keys

    def _next_id(self) -> str:
        self._seq += 1
        return f"bridge-{self._seq}"

    async def connect(self):
        """Connect to the OpenClaw gateway and complete the handshake."""
        logger.info("Connecting to OpenClaw gateway at %s", self.url)
        self._ws = await websockets.connect(self.url, max_size=10 * 1024 * 1024)

        # Read challenge
        challenge_raw = await asyncio.wait_for(self._ws.recv(), timeout=15)
        challenge = json.loads(challenge_raw)
        nonce = challenge.get("payload", {}).get("nonce", "")
        logger.info("Received challenge from gateway (nonce=%s...)", nonce[:8])

        # Build device identity and connect
        device_key = _load_or_create_device_key()
        dev_id = _device_id(device_key)
        client_id, client_mode, role = "gateway-client", "backend", "operator"
        scopes = ["operator.read", "operator.write"]
        plat = platform.system().lower()
        signed_at = int(time.time() * 1000)

        sig = _sign_device_payload(
            device_key, dev_id, client_id, client_mode, role,
            self.token, scopes, signed_at, nonce, plat,
        )

        connect_req = {
            "type": "req",
            "id": self._next_id(),
            "method": "connect",
            "params": {
                "minProtocol": 3, "maxProtocol": 3,
                "client": {
                    "id": client_id, "displayName": "Agentleh WhatsApp Bridge",
                    "version": "0.1.0", "platform": plat, "mode": client_mode,
                },
                "auth": {"token": self.token},
                "device": {
                    "id": dev_id, "publicKey": _public_key_b64(device_key),
                    "signature": sig, "signedAt": signed_at, "nonce": nonce,
                },
                "role": role, "scopes": scopes,
            },
        }

        await self._ws.send(json.dumps(connect_req))
        resp_raw = await asyncio.wait_for(self._ws.recv(), timeout=15)
        resp = json.loads(resp_raw)
        if resp.get("error"):
            raise ConnectionError(f"Gateway rejected connection: {resp['error']}")

        logger.info("Authenticated with OpenClaw gateway")
        self._read_task = asyncio.create_task(self._read_loop())

    async def ensure_connected(self):
        """Reconnect to the gateway if the WebSocket is closed."""
        if self._ws is not None and self._ws.close_code is None:
            return  # still connected

        logger.warning("Gateway disconnected, attempting reconnect...")
        # Clean up old connection
        if self._read_task:
            self._read_task.cancel()
            self._read_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Reconnect with exponential backoff
        for attempt in range(5):
            try:
                await self.connect()
                logger.info("Reconnected to gateway (attempt %d)", attempt + 1)
                return
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning("Reconnect attempt %d failed: %s (retry in %ds)", attempt + 1, e, wait)
                await asyncio.sleep(wait)

        raise ConnectionError("Failed to reconnect to gateway after 5 attempts")

    async def _read_loop(self):
        """Read incoming frames and route RPC responses to pending callers."""
        try:
            async for raw in self._ws:
                frame = json.loads(raw)
                if frame.get("type") == "res" and frame.get("id"):
                    fut = self._pending.pop(frame["id"], None)
                    if fut and not fut.done():
                        fut.set_result(frame)
        except websockets.ConnectionClosed:
            logger.warning("Gateway connection closed")
        except Exception as e:
            logger.error("Read loop error: %s", e)

    async def _send_request(self, method: str, params: dict) -> dict:
        req_id = self._next_id()
        req = {"type": "req", "id": req_id, "method": method, "params": params}
        fut = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps(req))
        try:
            return await asyncio.wait_for(fut, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"No response for {method} (id={req_id})")

    async def send_and_stream(
        self,
        text: str,
        on_message,
        sender: str = "",
        sender_name: str = "",
        role: str = "member",
        session_key: str = "",
        attachments: list[dict] | None = None,
    ):
        """Send a message to the agent and stream replies via on_message callback.

        Polls the session JSONL for assistant replies. Each reply with
        stopReason=stop is sent immediately via on_message(text). Keeps
        polling while the agent is still active (toolUse/toolResult entries).
        Returns when the agent has been idle for SETTLE_TIME seconds.

        attachments: list of {"mimeType": "...", "fileName": "...", "content": "<base64>"}
        """
        session_key = session_key or self.session_key
        tagged = f"From: {sender} ({sender_name}) [role: {role}]\n{text}"

        # Ensure we're connected (reconnect if needed)
        await self.ensure_connected()

        # Send chat.send
        params = {
            "sessionKey": session_key,
            "message": tagged,
            "idempotencyKey": str(uuid.uuid4()),
        }
        if attachments:
            params["attachments"] = attachments

        resp = await self._send_request("chat.send", params)
        if resp.get("error"):
            raise RuntimeError(f"chat.send rejected: {resp['error']}")

        logger.info("Message sent to agent (session=%s)", session_key)

        # Poll JSONL for replies
        since = time.time()
        deadline = since + POLL_TIMEOUT
        sessions_json = os.path.join(self.sessions_dir, "sessions.json")
        last_new_entry_at = 0.0
        seen_lines: set[str] = set()  # track which JSONL lines we've already processed
        got_reply = False

        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)

            session_file = _find_session_file(sessions_json, session_key, self.session_key)
            if not session_file:
                continue

            entries = _scan_jsonl(session_file, since)
            agent_active = False

            for entry in entries:
                msg = entry.get("message", {})
                line_key = f"{session_file}:{entry.get('_line', '')}"

                is_new = line_key not in seen_lines
                if is_new:
                    seen_lines.add(line_key)
                    last_new_entry_at = time.time()

                # Check if agent is mid-tool-call (even on already-seen entries,
                # because the state persists until a stop reply appears)
                if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                    agent_active = True
                if msg.get("role") == "toolResult":
                    agent_active = True
                # A stop reply after toolUse means the tool call is done
                if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                    agent_active = False

                # Deliver completed assistant replies (only once)
                if (is_new
                        and msg.get("role") == "assistant"
                        and msg.get("stopReason") == "stop"
                        and line_key not in self._claimed):
                    texts = []
                    for block in msg.get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            texts.append(block["text"])
                    if texts:
                        reply_text = "\n".join(texts).strip()
                        if reply_text:
                            self._claimed.add(line_key)
                            await on_message(reply_text)
                            got_reply = True
                            logger.info("Delivered reply (%d chars)", len(reply_text))

            # Cap claimed set size
            if len(self._claimed) > 1000:
                self._claimed = set(list(self._claimed)[-500:])

            # If agent is mid-tool-call, keep polling regardless of settle time
            if agent_active:
                continue

            # Only settle if we have at least one reply AND no new entries
            # for SETTLE_TIME seconds.
            if got_reply and last_new_entry_at > 0:
                if time.time() - last_new_entry_at >= SETTLE_TIME:
                    return

        if not got_reply:
            raise TimeoutError(f"Timeout waiting for agent reply (session={session_key})")

    async def close(self):
        if self._read_task:
            self._read_task.cancel()
        if self._ws:
            await self._ws.close()
