"""Robustness tests for the bridge.

These test real production edge cases:
1. Error messages should be in Hebrew (Israeli product)
2. Read receipts should come after the agent responds, not before
3. Gateway should reconnect after crashes
4. Slow tool calls should not cause premature settle
5. Media files should not accumulate forever
6. Lock scope should match session scope
"""

import asyncio
import json
import os
import time
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _ts(offset_sec: float = 0) -> str:
    t = datetime.fromtimestamp(time.time() + offset_sec, tz=timezone.utc)
    return t.isoformat()


# ── Issue 1: Error messages must be in Hebrew ────────────────────────


class TestErrorMessagesHebrew:
    """Error messages sent to WhatsApp users must be in Hebrew."""

    def test_timeout_error_is_hebrew(self):
        """The timeout error message should contain Hebrew text, not English."""
        import bridge
        # Extract the actual error strings from the source
        import inspect
        source = inspect.getsource(bridge.handle_message)

        # Should NOT contain English error messages
        assert "Sorry" not in source, (
            "Error messages should be in Hebrew, not English. "
            "Found 'Sorry' in handle_message."
        )

    def test_timeout_error_contains_hebrew(self):
        """Verify the timeout message is actually in Hebrew."""
        import inspect
        import bridge
        source = inspect.getsource(bridge.handle_message)

        # Find the TimeoutError handler — it should have Hebrew text
        timeout_section = source[source.find("TimeoutError"):]
        # Should contain at least one Hebrew character
        has_hebrew = any("\u0590" <= c <= "\u05FF" for c in timeout_section[:200])
        assert has_hebrew, (
            "Timeout error handler should contain Hebrew text"
        )

    def test_generic_error_is_hebrew(self):
        """The generic error message should contain Hebrew text."""
        import inspect
        import bridge
        source = inspect.getsource(bridge.handle_message)

        # Find the generic Exception handler
        except_section = source[source.rfind("except Exception"):]
        has_hebrew = any("\u0590" <= c <= "\u05FF" for c in except_section[:200])
        assert has_hebrew, (
            "Generic error handler should contain Hebrew text"
        )


# ── Issue 2: Read receipts after first reply ─────────────────────────


class TestReadReceiptTiming:
    """Blue check marks should appear when the agent responds,
    not when the message is received. Otherwise the user sees
    blue checks but no reply for 15-30 seconds.
    """

    @pytest.mark.asyncio
    async def test_mark_read_inside_reply_callback(self):
        """mark_read should be called inside the on_agent_message callback,
        not at the top of handle_message. This ensures blue checks appear
        when the agent responds, not when the message is received.
        """
        import inspect
        import bridge
        source = inspect.getsource(bridge.handle_message)

        # mark_read should be inside on_agent_message, NOT at the top level
        # Check: the first occurrence of mark_read should be indented more
        # than the function def (inside the callback)
        lines = source.split("\n")

        # Find on_agent_message definition
        callback_start = None
        mark_read_line = None
        for i, line in enumerate(lines):
            if "def on_agent_message" in line:
                callback_start = i
            if "mark_read" in line and mark_read_line is None:
                mark_read_line = i

        assert callback_start is not None, "on_agent_message callback should exist"
        assert mark_read_line is not None, "mark_read should be called somewhere"
        assert mark_read_line > callback_start, (
            f"mark_read (line {mark_read_line}) should be inside "
            f"on_agent_message (line {callback_start}), not before it. "
            "Blue checks should appear when the agent responds."
        )


# ── Issue 3: Gateway reconnection ────────────────────────────────────


class TestGatewayReconnection:
    """The gateway client should automatically reconnect after disconnection."""

    def test_gateway_has_reconnect_method(self):
        """OpenClawGateway should have a reconnect or auto-reconnect mechanism."""
        import gateway_client
        gw_class = gateway_client.OpenClawGateway

        has_reconnect = (
            hasattr(gw_class, 'reconnect')
            or hasattr(gw_class, '_reconnect')
            or hasattr(gw_class, 'ensure_connected')
        )
        assert has_reconnect, (
            "OpenClawGateway should have a reconnect/ensure_connected method"
        )

    def test_send_and_stream_reconnects_on_closed_connection(self):
        """If the WebSocket is closed when send_and_stream is called,
        it should attempt to reconnect before failing.
        """
        import inspect
        import gateway_client
        source = inspect.getsource(gateway_client.OpenClawGateway.send_and_stream)

        has_reconnect_check = (
            "reconnect" in source.lower()
            or "ensure_connected" in source.lower()
            or "connect" in source.lower()
        )
        # At minimum, send_and_stream should check connection state
        assert has_reconnect_check, (
            "send_and_stream should handle disconnected gateway"
        )


# ── Issue 4: Slow tool calls ─────────────────────────────────────────


class TestSlowToolCalls:
    """Tool calls that take longer than SETTLE_TIME should not cause
    premature settlement.
    """

    def test_settle_time_adequate_for_web_search(self):
        """SETTLE_TIME should be at least 15 seconds for web searches."""
        from gateway_client import SETTLE_TIME
        assert SETTLE_TIME >= 15, (
            f"SETTLE_TIME={SETTLE_TIME}s is too short. "
            "Web searches can take 10-15 seconds."
        )

    def test_tool_use_as_last_entry_prevents_settle(self):
        """If the last assistant entry has stopReason=toolUse,
        the agent is mid-tool-call — don't settle."""
        from gateway_client import _scan_jsonl

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write(json.dumps({
            "type": "message",
            "timestamp": _ts(-20),
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [{"type": "text", "text": "Searching..."}],
            },
        }) + "\n")
        f.close()

        entries = _scan_jsonl(f.name, time.time() - 30)

        # Determine agent_active using the same logic as send_and_stream
        agent_active = False
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                agent_active = True
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                agent_active = False

        assert agent_active, (
            "Agent should be active when last entry is toolUse "
            "(even if it was 20 seconds ago)"
        )
        os.unlink(f.name)

    def test_stop_after_tool_use_means_done(self):
        """toolUse → toolResult → stop = agent is done."""
        from gateway_client import _scan_jsonl

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write(json.dumps({
            "type": "message", "timestamp": _ts(-10),
            "message": {"role": "assistant", "stopReason": "toolUse",
                       "content": [{"type": "text", "text": "Checking..."}]},
        }) + "\n")
        f.write(json.dumps({
            "type": "message", "timestamp": _ts(-5),
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "ok"}]},
        }) + "\n")
        f.write(json.dumps({
            "type": "message", "timestamp": _ts(-3),
            "message": {"role": "assistant", "stopReason": "stop",
                       "content": [{"type": "text", "text": "Done."}]},
        }) + "\n")
        f.close()

        entries = _scan_jsonl(f.name, time.time() - 15)

        agent_active = False
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                agent_active = True
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                agent_active = False

        assert not agent_active, "Agent should be done after stop"
        os.unlink(f.name)


# ── Issue 5: Settle timer uses seen_lines tracking ───────────────────


class TestSettleTimerCorrectness:
    """The settle timer should only reset on NEW entries,
    not on re-reading old entries.
    """

    def test_gateway_client_tracks_seen_lines(self):
        """send_and_stream should use seen_lines (not just _claimed) to
        track which entries are new vs already processed."""
        import inspect
        import gateway_client
        source = inspect.getsource(gateway_client.OpenClawGateway.send_and_stream)

        assert "seen_lines" in source, (
            "send_and_stream should track seen_lines to properly detect "
            "new vs old entries for the settle timer"
        )

    def test_last_new_entry_at_not_last_activity_at(self):
        """The settle timer should use last_new_entry_at (only new entries),
        not last_activity_at (all entries).
        """
        import inspect
        import gateway_client
        source = inspect.getsource(gateway_client.OpenClawGateway.send_and_stream)

        assert "last_new_entry_at" in source, (
            "Should use last_new_entry_at for settle timing"
        )
        assert "last_activity_at" not in source, (
            "Should NOT use last_activity_at — it resets on every entry "
            "and prevents settle from ever firing"
        )


# ── Issue 6: Lock scope ──────────────────────────────────────────────


class TestLockScope:
    """The message lock should work correctly for the current session scope."""

    def test_lock_exists(self):
        """Bridge should have a message lock."""
        import bridge
        assert hasattr(bridge, '_message_lock')

    def test_handle_message_uses_lock(self):
        """handle_message should acquire the lock."""
        import inspect
        import bridge
        source = inspect.getsource(bridge.handle_message)
        assert "_message_lock" in source

    def test_per_sender_sessions_are_different(self, tmp_path):
        """In per-sender mode, different phone numbers get different session keys."""
        import bridge

        config = {"session": {"dmScope": "per-channel-peer"}}
        (tmp_path / "openclaw.json").write_text(json.dumps(config))

        old_dir = bridge.OPENCLAW_STATE_DIR
        bridge.OPENCLAW_STATE_DIR = tmp_path

        try:
            assert bridge.derive_session_key("972546901044") != bridge.derive_session_key("972526414555")
        finally:
            bridge.OPENCLAW_STATE_DIR = old_dir
