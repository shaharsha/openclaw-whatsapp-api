"""Tests that reproduce the exact timeout/settle issues found in production logs.

Each test models a real scenario from the bridge logs (2026-04-03):
1. Settle fires too early during tool calls (web search takes >6s)
2. Multiple concurrent messages cause racing JSONL polls
3. Reply delivered after bridge already timed out
"""

import asyncio
import json
import os
import time
import tempfile

import pytest

from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gateway_client import _scan_jsonl, SETTLE_TIME


# ── Helpers ──────────────────────────────────────────────────────────


def _ts(offset_sec: float = 0) -> str:
    t = datetime.fromtimestamp(time.time() + offset_sec, tz=timezone.utc)
    return t.isoformat()


def _assistant(text: str, stop: str = "stop", offset: float = 0) -> str:
    return json.dumps({
        "type": "message",
        "timestamp": _ts(offset),
        "message": {
            "role": "assistant",
            "stopReason": stop,
            "content": [{"type": "text", "text": text}],
        },
    })


def _tool_use(text: str = "Let me check...", offset: float = 0) -> str:
    return json.dumps({
        "type": "message",
        "timestamp": _ts(offset),
        "message": {
            "role": "assistant",
            "stopReason": "toolUse",
            "content": [
                {"type": "text", "text": text},
                {"type": "toolCall", "toolName": "web_fetch"},
            ],
        },
    })


def _tool_result(result: str = '{"status": 200}', offset: float = 0) -> str:
    return json.dumps({
        "type": "message",
        "timestamp": _ts(offset),
        "message": {
            "role": "toolResult",
            "toolName": "web_fetch",
            "content": [{"type": "text", "text": result}],
        },
    })


def _user(text: str = "hello", offset: float = 0) -> str:
    return json.dumps({
        "type": "message",
        "timestamp": _ts(offset),
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    })


def _write_jsonl(lines: list[str]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


def _append_to_jsonl(path: str, line: str):
    with open(path, "a") as f:
        f.write(line + "\n")


# ── Scenario 1: Settle too early during tool call ────────────────────


class TestSettleDuringToolCall:
    """Reproduce: Agent says "let me check", starts web_fetch, no JSONL entries
    for 10+ seconds (tool executing), bridge settles and exits.

    From logs at 02:27:15 — agent researching online, bridge timed out after
    6 seconds, actual response came 10 seconds later.
    """

    def test_settle_time_must_be_longer_than_tool_calls(self):
        """SETTLE_TIME should be at least 15 seconds to handle web searches."""
        assert SETTLE_TIME >= 15, (
            f"SETTLE_TIME is {SETTLE_TIME}s — too short for web search tool calls "
            f"which can take 10-15 seconds. Should be >= 15s."
        )

    def test_tool_use_entry_prevents_settling(self):
        """If the last JSONL entry is toolUse, the agent is active — don't settle."""
        path = _write_jsonl([
            _user("search online for openclaw best practices"),
            _tool_use("Let me research this...", offset=0.1),
        ])

        entries = _scan_jsonl(path, time.time() - 5)

        # The polling logic should detect agent_active = True
        agent_active = False
        for entry in entries:
            msg = entry.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                agent_active = True
            if msg.get("role") == "toolResult":
                agent_active = True

        assert agent_active, "Agent should be detected as active after toolUse entry"
        os.unlink(path)

    def test_no_entries_after_tool_use_still_active(self):
        """If we saw a toolUse but no toolResult or stop yet, agent is still working.

        This is the critical scenario: the JSONL has toolUse as the last entry,
        and on the NEXT poll nothing new has appeared (tool is executing).
        The bridge should NOT settle.
        """
        path = _write_jsonl([
            _user("search online"),
            _tool_use("Searching...", offset=0.1),
        ])

        # First scan — finds toolUse
        entries1 = _scan_jsonl(path, time.time() - 5)
        has_tool_use = any(
            e.get("message", {}).get("stopReason") == "toolUse"
            for e in entries1
        )
        assert has_tool_use

        # Second scan (simulating next poll) — same entries, nothing new
        entries2 = _scan_jsonl(path, time.time() - 5)

        # The agent_active flag should still be True because the last
        # assistant entry has stopReason=toolUse with no following stop
        last_assistant_stop = None
        for e in entries2:
            msg = e.get("message", {})
            if msg.get("role") == "assistant":
                last_assistant_stop = msg.get("stopReason")

        assert last_assistant_stop == "toolUse", (
            "Last assistant entry should still be toolUse — agent is mid-tool-call"
        )
        os.unlink(path)

    def test_tool_result_followed_by_stop_means_done(self):
        """After toolUse → toolResult → stop, the agent is truly done."""
        path = _write_jsonl([
            _user("search online"),
            _tool_use("Searching...", offset=0.1),
            _tool_result('{"results": "found"}', offset=5),
            _assistant("Here's what I found.", offset=6),
        ])

        entries = _scan_jsonl(path, time.time() - 10)

        # Should have a final stop reply
        final_replies = [
            e for e in entries
            if e.get("message", {}).get("role") == "assistant"
            and e.get("message", {}).get("stopReason") == "stop"
        ]
        assert len(final_replies) == 1

        # Agent should NOT be active (the last entry is stop, not toolUse)
        last_assistant_stop = None
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant":
                last_assistant_stop = msg.get("stopReason")

        assert last_assistant_stop == "stop", "Agent should be done after stop"
        os.unlink(path)


# ── Scenario 2: Multiple rapid messages ──────────────────────────────


class TestRapidMessages:
    """Reproduce: User sends 4 messages in 2 minutes while agent is processing.

    From logs at 02:24-02:27 — each message starts a concurrent JSONL poll.
    The REAL issue: concurrent tasks share `self._claimed`, so Task A claims
    reply 1, then Task B starts polling with a LATER `since` timestamp
    and correctly finds reply 2 (not claimed). This works IF each task has
    a unique `since` that filters to only its own reply.

    The problem occurs when Task A has `since` that includes reply 2's
    timestamp, claims reply 2, and Task B never finds anything → timeout.
    """

    def test_dedup_prevents_double_delivery(self):
        """Same JSONL line should never be delivered twice even across concurrent polls."""
        path = _write_jsonl([
            _user("msg 1"),
            _assistant("reply to msg 1", offset=0.1),
        ])

        entries = _scan_jsonl(path, time.time() - 5)

        # Simulate two concurrent polls claiming the same reply
        claimed = set()
        delivered = []

        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                key = f"{path}:{e['_line']}"
                if key not in claimed:
                    claimed.add(key)
                    delivered.append(e)

        # Second poll — same entries
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                key = f"{path}:{e['_line']}"
                if key not in claimed:
                    claimed.add(key)
                    delivered.append(e)

        assert len(delivered) == 1, "Reply should only be delivered once"
        os.unlink(path)

    def test_concurrent_tasks_steal_each_others_replies(self):
        """CRITICAL BUG: Task A (since=T0) polls and claims reply 2 (at T0+5)
        before Task B (since=T0+3) gets a chance to find it.

        This is the root cause of the 02:27:33 timeout — a task from 5 minutes
        earlier was still polling and claimed a reply meant for a later message.
        """
        now = time.time()

        # Timeline:
        # T0:    User sends msg 1
        # T0+1:  Agent replies to msg 1
        # T0+3:  User sends msg 2
        # T0+5:  Agent replies to msg 2
        path = _write_jsonl([
            _user("msg 1", offset=-10),
            _assistant("reply 1", offset=-9),
            _user("msg 2", offset=-7),
            _assistant("reply 2", offset=-5),
        ])

        # Task A started at T0, so since=T0-10
        # It scans and finds BOTH reply 1 and reply 2
        claimed = set()
        entries_a = _scan_jsonl(path, now - 10)

        replies_for_a = []
        for e in entries_a:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                key = f"{path}:{e['_line']}"
                if key not in claimed:
                    claimed.add(key)
                    replies_for_a.append(e["message"]["content"][0]["text"])

        # Task A greedily claimed BOTH replies!
        assert len(replies_for_a) == 2, (
            "Task A sees both replies (this IS the bug — it should only claim "
            "the first one, but there's no way to know which reply is 'mine')"
        )

        # Task B started at T0+3, so since=T0-7
        entries_b = _scan_jsonl(path, now - 7)

        replies_for_b = []
        for e in entries_b:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                key = f"{path}:{e['_line']}"
                if key not in claimed:
                    claimed.add(key)
                    replies_for_b.append(e["message"]["content"][0]["text"])

        # Task B finds NOTHING — both replies were already claimed by Task A
        # This causes Task B to timeout after POLL_TIMEOUT seconds
        assert len(replies_for_b) == 0, (
            "Task B finds nothing because Task A claimed its reply — THIS IS THE BUG"
        )

        os.unlink(path)

    def test_sequential_replies_each_delivered_once(self):
        """Multiple user→assistant exchanges should each be delivered exactly once."""
        path = _write_jsonl([
            _user("msg 1"),
            _assistant("reply 1", offset=0.1),
            _user("msg 2", offset=1),
            _assistant("reply 2", offset=1.1),
            _user("msg 3", offset=2),
            _assistant("reply 3", offset=2.1),
        ])

        entries = _scan_jsonl(path, time.time() - 5)

        claimed = set()
        delivered = []
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                key = f"{path}:{e['_line']}"
                if key not in claimed:
                    claimed.add(key)
                    texts = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
                    delivered.append("\n".join(texts).strip())

        assert len(delivered) == 3
        assert delivered == ["reply 1", "reply 2", "reply 3"]
        os.unlink(path)


# ── Scenario 3: Reply arrives after timeout ──────────────────────────


class TestReplyAfterTimeout:
    """Reproduce: Bridge times out, sends error to user, then agent's actual
    reply arrives and gets delivered as a second message.

    From logs: timeout at 02:27:33, actual delivery at 02:27:43.
    """

    def test_late_reply_in_jsonl(self):
        """If a reply appears in JSONL after the poll started, it should be findable."""
        path = _write_jsonl([
            _user("research online"),
            _tool_use("Researching...", offset=0.1),
        ])

        # First scan — no stop reply yet
        entries1 = _scan_jsonl(path, time.time() - 5)
        stop_replies = [
            e for e in entries1
            if e.get("message", {}).get("stopReason") == "stop"
        ]
        assert len(stop_replies) == 0

        # Simulate tool completing and response appearing
        _append_to_jsonl(path, _tool_result('{"data": "result"}', offset=10))
        _append_to_jsonl(path, _assistant("Here's what I found from research.", offset=11))

        # Second scan — now has the reply
        entries2 = _scan_jsonl(path, time.time() - 15)
        stop_replies = [
            e for e in entries2
            if e.get("message", {}).get("stopReason") == "stop"
        ]
        assert len(stop_replies) == 1
        assert "research" in stop_replies[0]["message"]["content"][0]["text"]
        os.unlink(path)


# ── Scenario 4: Agent active detection edge cases ────────────────────


class TestAgentActiveDetection:
    """Test that agent_active is correctly determined from JSONL state."""

    def test_agent_active_after_tool_use_no_result(self):
        """toolUse without toolResult = agent is active."""
        path = _write_jsonl([
            _user("do something"),
            _tool_use("Working on it...", offset=0.1),
        ])
        entries = _scan_jsonl(path, time.time() - 5)

        # Walk entries to determine state
        last_was_tool_use = False
        has_stop_after_tool = False
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                last_was_tool_use = True
            if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                if last_was_tool_use:
                    has_stop_after_tool = True

        assert last_was_tool_use and not has_stop_after_tool
        os.unlink(path)

    def test_agent_not_active_after_simple_reply(self):
        """Simple reply with no tool calls = agent is done."""
        path = _write_jsonl([
            _user("hi"),
            _assistant("hello!", offset=0.1),
        ])
        entries = _scan_jsonl(path, time.time() - 5)

        has_tool_activity = any(
            e.get("message", {}).get("stopReason") == "toolUse"
            or e.get("message", {}).get("role") == "toolResult"
            for e in entries
        )
        assert not has_tool_activity
        os.unlink(path)

    def test_agent_active_during_multi_tool_chain(self):
        """Agent does tool1 → result1 → tool2 (still active)."""
        path = _write_jsonl([
            _user("complex task"),
            _tool_use("Step 1...", offset=0.1),
            _tool_result("result 1", offset=1),
            _assistant("Got step 1, now step 2:", stop="toolUse", offset=1.1),
        ])
        entries = _scan_jsonl(path, time.time() - 5)

        # Last assistant entry has stopReason=toolUse
        last_stop = None
        for e in entries:
            msg = e.get("message", {})
            if msg.get("role") == "assistant":
                last_stop = msg.get("stopReason")

        assert last_stop == "toolUse", "Agent should still be active (mid second tool call)"
        os.unlink(path)
