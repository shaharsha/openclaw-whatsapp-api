"""Integration tests for the bridge polling loop.

These tests simulate the full send_and_stream flow by creating real JSONL files
and writing entries to them on a timer (simulating the agent writing responses).
They test the actual polling logic, timing, settle behavior, and concurrency.
"""

import asyncio
import json
import os
import tempfile
import time

import pytest

from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gateway_client
from gateway_client import (
    OpenClawGateway,
    _find_session_file,
    _scan_jsonl,
    POLL_INTERVAL,
    POLL_TIMEOUT,
    SETTLE_TIME,
)

# Use faster timings for tests
TEST_SETTLE_TIME = 3.0
TEST_POLL_INTERVAL = 0.3


# ── Helpers ──────────────────────────────────────────────────────────


def _async_append(lst: list):
    """Create an async callback that appends to a list."""
    async def _cb(text: str):
        lst.append(text)
    return _cb


async def _async_noop(text: str):
    pass


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


def _tool_use(text: str = "Checking...", offset: float = 0) -> str:
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


def _tool_result(result: str = '{"ok": true}', offset: float = 0) -> str:
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


class FakeGateway:
    """A gateway that simulates the JSONL polling without a real WebSocket.

    Provides send_and_stream with the real polling logic but a fake chat.send
    that just writes a user entry to the JSONL.
    """

    def __init__(self, sessions_dir: str):
        self.sessions_dir = sessions_dir
        self.session_key = "main"
        self._claimed: set[str] = set()

    def _setup_sessions_json(self, jsonl_path: str):
        sj = os.path.join(self.sessions_dir, "sessions.json")
        with open(sj, "w") as f:
            json.dump({
                f"agent:{self.session_key}:{self.session_key}": {
                    "sessionFile": jsonl_path,
                }
            }, f)

    async def send_and_stream(
        self,
        text: str,
        on_message,
        session_key: str = "",
        settle_time: float = TEST_SETTLE_TIME,
        poll_interval: float = TEST_POLL_INTERVAL,
        poll_timeout: float = 30.0,
        **kwargs,
    ):
        """Real polling logic from OpenClawGateway, but without WebSocket."""
        session_key = session_key or self.session_key
        since = time.time()
        deadline = since + poll_timeout
        sessions_json = os.path.join(self.sessions_dir, "sessions.json")
        last_new_entry_at = 0.0
        seen_lines: set[str] = set()
        got_reply = False

        while time.time() < deadline:
            await asyncio.sleep(poll_interval)

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

                if msg.get("role") == "assistant" and msg.get("stopReason") == "toolUse":
                    agent_active = True
                if msg.get("role") == "toolResult":
                    agent_active = True
                if msg.get("role") == "assistant" and msg.get("stopReason") == "stop":
                    agent_active = False

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

            if len(self._claimed) > 1000:
                self._claimed = set(list(self._claimed)[-500:])

            if agent_active:
                continue

            if got_reply and last_new_entry_at > 0:
                if time.time() - last_new_entry_at >= settle_time:
                    return

        if not got_reply:
            raise TimeoutError(f"Timeout (session={session_key})")


# ── Test: Simple reply ───────────────────────────────────────────────


class TestSimpleReplyIntegration:
    """Agent replies immediately with a simple text message."""

    @pytest.mark.asyncio
    async def test_simple_reply_delivered(self, tmp_path):
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        # Pre-write: user message + agent reply
        with open(jsonl_path, "w") as f:
            f.write(_user("hi") + "\n")
            f.write(_assistant("hello!", offset=0.1) + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        replies = []
        await gw.send_and_stream("hi", on_message=_async_append(replies))

        assert replies == ["hello!"]

    @pytest.mark.asyncio
    async def test_reply_arrives_after_delay(self, tmp_path):
        """Agent takes 2 seconds to respond — poll should find it."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("think about this") + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        # Write the reply after 2 seconds
        async def delayed_reply():
            await asyncio.sleep(2)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("I thought about it.") + "\n")

        replies = []
        task = asyncio.create_task(delayed_reply())
        await gw.send_and_stream("think", on_message=_async_append(replies))
        await task

        assert replies == ["I thought about it."]


# ── Test: Tool call flow ─────────────────────────────────────────────


class TestToolCallIntegration:
    """Agent does toolUse → toolResult → final reply."""

    @pytest.mark.asyncio
    async def test_tool_call_waits_for_final_reply(self, tmp_path):
        """Should keep polling through toolUse and toolResult, only settle after stop."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("check weather") + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def simulate_tool_call():
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_use("Let me check the weather...") + "\n")
            await asyncio.sleep(2)  # tool executing
            with open(jsonl_path, "a") as f:
                f.write(_tool_result('{"temp": "15°C"}') + "\n")
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("The weather is 15°C and sunny.") + "\n")

        replies = []
        task = asyncio.create_task(simulate_tool_call())
        await gw.send_and_stream("weather", on_message=_async_append(replies))
        await task

        assert replies == ["The weather is 15°C and sunny."]

    @pytest.mark.asyncio
    async def test_slow_tool_call_doesnt_settle_early(self, tmp_path):
        """Tool call takes 5 seconds — should NOT settle during execution."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("research") + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def simulate_slow_tool():
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_use("Researching...") + "\n")
            # Tool takes 5 seconds (longer than old SETTLE_TIME of 6s)
            await asyncio.sleep(5)
            with open(jsonl_path, "a") as f:
                f.write(_tool_result('{"data": "found"}') + "\n")
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("Found the answer.") + "\n")

        replies = []
        task = asyncio.create_task(simulate_slow_tool())
        await gw.send_and_stream("research", on_message=_async_append(replies))
        await task

        assert replies == ["Found the answer."], (
            f"Expected final reply, got {replies}. "
            "If empty, settle fired during tool execution."
        )

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls(self, tmp_path):
        """Agent does two sequential tool calls, each with its own result."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("weather in Paris and London") + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def simulate_multi_tool():
            await asyncio.sleep(1)
            # First tool call
            with open(jsonl_path, "a") as f:
                f.write(_tool_use("Checking Paris...") + "\n")
            await asyncio.sleep(2)
            with open(jsonl_path, "a") as f:
                f.write(_tool_result('{"paris": "9°C"}') + "\n")
            await asyncio.sleep(0.5)
            # Second tool call
            with open(jsonl_path, "a") as f:
                f.write(json.dumps({
                    "type": "message",
                    "timestamp": _ts(),
                    "message": {
                        "role": "assistant",
                        "stopReason": "toolUse",
                        "content": [
                            {"type": "text", "text": "Now checking London..."},
                            {"type": "toolCall", "toolName": "web_fetch"},
                        ],
                    },
                }) + "\n")
            await asyncio.sleep(2)
            with open(jsonl_path, "a") as f:
                f.write(_tool_result('{"london": "7°C"}') + "\n")
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("Paris: 9°C, London: 7°C.") + "\n")

        replies = []
        task = asyncio.create_task(simulate_multi_tool())
        await gw.send_and_stream("weather", on_message=_async_append(replies))
        await task

        assert replies == ["Paris: 9°C, London: 7°C."]


# ── Test: Concurrent message serialization ───────────────────────────


class TestConcurrentMessageSerialization:
    """Test that the asyncio.Lock in handle_message prevents reply stealing."""

    @pytest.mark.asyncio
    async def test_serialized_messages_each_get_own_reply(self, tmp_path):
        """Two messages sent rapidly — with serialization, each gets its own reply.

        The lock ensures msg2 doesn't start polling until msg1 is done.
        The agent writes replies reactively (after the message is sent).
        """
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write("")  # empty

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        lock = asyncio.Lock()
        all_replies: dict[str, list[str]] = {"msg1": [], "msg2": []}
        msg_count = 0

        async def watch_and_reply():
            """Watch the JSONL for user messages and write replies after a delay."""
            nonlocal msg_count
            seen = 0
            for _ in range(100):  # poll for up to ~30s
                await asyncio.sleep(0.3)
                try:
                    with open(jsonl_path) as f:
                        lines = [l.strip() for l in f if l.strip()]
                    # Count user messages
                    user_msgs = sum(1 for l in lines if '"role": "user"' in l or '"role":"user"' in l)
                    if user_msgs > seen:
                        seen = user_msgs
                        await asyncio.sleep(1)  # agent "thinking"
                        with open(jsonl_path, "a") as f:
                            f.write(_assistant(f"reply to msg {seen}") + "\n")
                        msg_count += 1
                except FileNotFoundError:
                    continue
                if msg_count >= 2:
                    break

        async def send_msg(msg_id: str, msg_num: int):
            async with lock:
                # Write user message to JSONL (simulates chat.send)
                with open(jsonl_path, "a") as f:
                    f.write(_user(f"msg {msg_num}") + "\n")
                replies = []
                await gw.send_and_stream(
                    msg_id,
                    on_message=_async_append(replies),
                )
                all_replies[msg_id] = replies

        agent_task = asyncio.create_task(watch_and_reply())

        await asyncio.gather(
            send_msg("msg1", 1),
            send_msg("msg2", 2),
        )
        await agent_task

        assert len(all_replies["msg1"]) == 1, f"msg1 replies: {all_replies['msg1']}"
        assert len(all_replies["msg2"]) == 1, f"msg2 replies: {all_replies['msg2']}"
        assert all_replies["msg1"][0] == "reply to msg 1"
        assert all_replies["msg2"][0] == "reply to msg 2"

    @pytest.mark.asyncio
    async def test_without_lock_replies_get_stolen(self, tmp_path):
        """WITHOUT serialization, concurrent tasks steal each other's replies.

        This test proves the bug exists when there's no lock.
        Both tasks start at nearly the same time and share a _claimed set.
        """
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write("")

        shared_claimed: set[str] = set()

        gw_a = FakeGateway(sessions_dir)
        gw_a._claimed = shared_claimed
        gw_a._setup_sessions_json(jsonl_path)

        gw_b = FakeGateway(sessions_dir)
        gw_b._claimed = shared_claimed  # shared — this causes the bug

        replies_a = []
        replies_b = []

        async def write_both_replies():
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_user("msg 1") + "\n")
                f.write(_assistant("reply to msg 1") + "\n")
                f.write(_user("msg 2") + "\n")
                f.write(_assistant("reply to msg 2") + "\n")

        async def task_a():
            await gw_a.send_and_stream(
                "msg1", on_message=_async_append(replies_a),
            )

        async def task_b():
            await asyncio.sleep(0.1)
            try:
                await gw_b.send_and_stream(
                    "msg2", on_message=_async_append(replies_b),
                    poll_timeout=TEST_SETTLE_TIME + 5,
                )
            except TimeoutError:
                pass

        writer = asyncio.create_task(write_both_replies())
        await asyncio.gather(task_a(), task_b())
        await writer

        # Task A greedily claimed both replies (the bug)
        assert len(replies_a) == 2, (
            f"Task A should claim both replies (the bug), got {replies_a}"
        )
        assert len(replies_b) == 0, (
            f"Task B should get nothing (the bug), got {replies_b}"
        )


# ── Test: Settle timing ──────────────────────────────────────────────


class TestSettleTiming:
    """Test that settle fires at the right time — not too early, not too late."""

    @pytest.mark.asyncio
    async def test_settles_after_idle_period(self, tmp_path):
        """After a reply, should wait settle_time then return."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write("")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def write_reply():
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_user("hi") + "\n")
                f.write(_assistant("hello!") + "\n")

        start = time.time()
        replies = []
        task = asyncio.create_task(write_reply())
        await gw.send_and_stream("hi", on_message=_async_append(replies))
        await task
        elapsed = time.time() - start

        assert replies == ["hello!"]
        assert elapsed >= TEST_SETTLE_TIME, (
            f"Settled too fast ({elapsed:.1f}s < {TEST_SETTLE_TIME}s)"
        )
        assert elapsed < TEST_SETTLE_TIME + 3, (
            f"Settled too slow ({elapsed:.1f}s)"
        )

    @pytest.mark.asyncio
    async def test_no_reply_waits_for_timeout(self, tmp_path):
        """If no reply ever appears, should wait full timeout then raise."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("hello") + "\n")
            # No reply

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        with pytest.raises(TimeoutError):
            await gw.send_and_stream(
                "hello", on_message=_async_noop, poll_timeout=3.0,
            )


# ── Test: Dedup within a single task ─────────────────────────────────


class TestDedupWithinTask:
    """Each reply should be delivered exactly once even though the poll
    re-reads the entire JSONL on every iteration."""

    @pytest.mark.asyncio
    async def test_same_reply_not_delivered_twice(self, tmp_path):
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write("")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def write_reply():
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_user("hi") + "\n")
                f.write(_assistant("hello!") + "\n")

        replies = []
        task = asyncio.create_task(write_reply())
        await gw.send_and_stream("hi", on_message=_async_append(replies))
        await task

        # The poll runs multiple times during SETTLE_TIME, but the reply
        # should only appear once
        assert replies == ["hello!"]
        assert len(replies) == 1

    @pytest.mark.asyncio
    async def test_multiple_replies_each_delivered_once(self, tmp_path):
        """Agent sends two separate stop replies (multi-turn) — each once."""
        sessions_dir = str(tmp_path)
        jsonl_path = str(tmp_path / "session.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(_user("complex request") + "\n")

        gw = FakeGateway(sessions_dir)
        gw._setup_sessions_json(jsonl_path)

        async def simulate():
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_use("Step 1...") + "\n")
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_result("result 1") + "\n")
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("Step 1 done.") + "\n")
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_use("Step 2...") + "\n")
            await asyncio.sleep(1)
            with open(jsonl_path, "a") as f:
                f.write(_tool_result("result 2") + "\n")
            await asyncio.sleep(0.5)
            with open(jsonl_path, "a") as f:
                f.write(_assistant("All done!") + "\n")

        replies = []
        task = asyncio.create_task(simulate())
        await gw.send_and_stream("complex", on_message=_async_append(replies))
        await task

        assert replies == ["Step 1 done.", "All done!"]
