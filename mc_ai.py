#!/usr/bin/env python3
"""
MeshCore -> Gemini channel bot

- Connects to a MeshCore device over WiFi/TCP
- Listens on a specific channel (e.g. #avl-ai)
- Triggers on "!ai ..." appearing either at start of message OR after a "NAME: " prefix
- Replies in-channel, prefixed with an @mention of the sender (e.g. "@iX-HTv4-1 ...")
- De-dupes duplicate inbound packets within a short window to avoid double replies

Env vars:
  GEMINI_API_KEY                (required) Gemini API key
  MESHCORE_HOST                 (required) MeshCore TCP host/IP
  MESHCORE_PORT                 (default: 5000)
  MESHCORE_CHANNEL_NAME         (default: #avl-ai)  (accepts with/without leading #)
  CHANNEL_SCAN_MAX              (default: 16)
  AI_TRIGGER                    (default: !ai)  (NOTE: in bash use single quotes: export AI_TRIGGER='!ai')
  GEMINI_MODEL                  (default: gemini-3-flash-preview)
  MAX_REPLY_CHARS               (default: 180)  chunk size per mesh message
  HISTORY_TURNS                 (default: 6)    turns to keep per channel
  DEBUG                         (default: 0)    set to 1 for verbose
  DEDUPE_WINDOW_S               (default: 3.0)  seconds to treat identical inbound as duplicate
"""

import asyncio
import os
import re
import time
from collections import deque
from typing import Deque, Dict, List, Tuple

from google import genai
from meshcore import MeshCore, EventType


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return default if not v else int(v)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return default if not v else float(v)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return default if not v else v


def normalize_channel_name(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("#"):
        n = n[1:]
    return n.strip().lower()


def chunk_text(text: str, max_len: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    cur = ""
    # split preserving whitespace tokens
    for tok in re.split(r"(\s+)", text):
        if len(cur) + len(tok) <= max_len:
            cur += tok
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = tok
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


async def resolve_channel_idx(mesh: MeshCore, channel_name: str, max_channels: int = 16) -> int:
    want = normalize_channel_name(channel_name)
    for idx in range(max_channels):
        ev = await mesh.commands.get_channel(idx)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if not isinstance(payload, dict):
            continue

        # observed payload key: channel_name
        got_raw = payload.get("channel_name") or payload.get("name") or payload.get("chan_name") or ""
        got = normalize_channel_name(str(got_raw))
        if got == want:
            return idx

    raise RuntimeError(f"Channel '{channel_name}' not found in first {max_channels} channel slots")


class ChannelGeminiBot:
    def __init__(
        self,
        mesh: MeshCore,
        gemini_client: genai.Client,
        channel_idx: int,
        channel_label: str,
        trigger: str,
        model: str,
        max_reply_chars: int,
        history_turns: int,
        dedupe_window_s: float,
        debug: bool,
    ):
        self.mesh = mesh
        self.gemini = gemini_client
        self.channel_idx = channel_idx
        self.channel_label = channel_label
        self.trigger = trigger
        self.model = model
        self.max_reply_chars = max_reply_chars
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s

        # Accept trigger at start or after "NAME: "
        # We apply this to the BODY (after optional sender prefix split)
        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)

        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            channel_idx: deque(maxlen=history_turns * 2)
        }
        self._lock = asyncio.Lock()

        # De-dupe recent inbound messages
        self._seen_ts: Dict[Tuple[int, int, str], float] = {}  # key -> time.time()

    @staticmethod
    def split_sender_and_body(text: str) -> Tuple[str, str]:
        """
        If message looks like 'NAME: something', return ('NAME', 'something').
        Otherwise return ('', original_text).
        """
        t = (text or "").strip()
        if ": " in t:
            name, body = t.split(": ", 1)
            name = name.strip()
            body = body.strip()
            # Basic guard: treat very long prefixes as not-a-name
            if name and len(name) <= 40:
                return name, body
        return "", t

    @staticmethod
    def format_reply(sender: str, msg: str) -> str:
        msg = (msg or "").strip()
        if not msg:
            return ""
        return f"@{sender} {msg}" if sender else msg

    def extract_after_trigger(self, body: str) -> str:
        """
        Find trigger token in body and return text after it.
        Examples:
          "!ai ping" -> "ping"
          "foo !ai ping" -> "ping"  (rare but supported)
        """
        b = (body or "").strip()
        m = self.trigger_re.search(b)
        if not m:
            return ""
        # find the actual "!ai" occurrence start in the match and slice from its end
        # easiest: locate the trigger case-insensitively and slice after first occurrence
        idx = b.lower().find(self.trigger.lower())
        if idx < 0:
            return ""
        return b[idx + len(self.trigger) :].strip(" \t:,-")

    def _dedupe_drop(self, payload: dict, body: str) -> bool:
        """
        Returns True if this message is a duplicate and should be dropped.
        """
        ch = payload.get("channel_idx")
        st = payload.get("sender_timestamp")
        if not isinstance(ch, int) or not isinstance(st, int):
            return False

        key = (ch, st, body)
        now = time.time()

        # purge old
        for k, t0 in list(self._seen_ts.items()):
            if now - t0 > self.dedupe_window_s:
                self._seen_ts.pop(k, None)

        if key in self._seen_ts:
            if self.debug:
                print(f"[DBG] duplicate dropped key={key}")
            return True

        self._seen_ts[key] = now
        return False

    def build_prompt(self, user_text: str) -> str:
        hist = list(self.history[self.channel_idx])
        lines = [
            "You are a concise assistant replying over a low-bandwidth MeshCore channel.",
            "Keep replies short and directly useful. Prefer 1-3 sentences.",
            "If uncertain, say so briefly.",
            "",
            f"Channel: {self.channel_label}",
            "",
            "Conversation:",
        ]
        for role, msg in hist:
            lines.append(f"{role}: {msg}")
        lines.append(f"user: {user_text}")
        lines.append("assistant:")
        return "\n".join(lines)

    async def call_gemini(self, prompt: str) -> str:
        resp = self.gemini.models.generate_content(model=self.model, contents=prompt)
        txt = getattr(resp, "text", None)
        if not txt or not str(txt).strip():
            return "I couldn’t generate a response."
        return str(txt).strip()

    async def on_channel_msg(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        # only our channel
        if p.get("channel_idx") != self.channel_idx:
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)

        # drop duplicates (common to see dupes on some setups)
        if self._dedupe_drop(p, body):
            return

        if self.debug:
            print(f"[DBG] target-channel msg payload={p}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._lock:
            # ping
            if user.lower() == "ping":
                out = self.format_reply(sender, "pong")
                if out:
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)
                return

            # Gemini
            self.history[self.channel_idx].append(("user", user))
            prompt = self.build_prompt(user)

            try:
                answer = await self.call_gemini(prompt)
            except Exception as e:
                answer = f"Gemini error: {e}"

            self.history[self.channel_idx].append(("assistant", answer))

            parts = chunk_text(answer, self.max_reply_chars)
            if not parts:
                return

            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_reply(sender, msg)
                if out:
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)


async def main() -> None:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise SystemExit("Missing GEMINI_API_KEY")
    host = env_str("MESHCORE_HOST", "")
    if not host:
        raise SystemExit("Missing MESHCORE_HOST")

    port = env_int("MESHCORE_PORT", 5000)
    channel_name = env_str("MESHCORE_CHANNEL_NAME", "#avl-ai")
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    model = env_str("GEMINI_MODEL", "gemini-3-flash-preview")
    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")

    mesh = await MeshCore.create_tcp(host, port, auto_reconnect=True)
    await mesh.start_auto_message_fetching()

    chan_idx = await resolve_channel_idx(mesh, channel_name, max_channels=scan_max)

    # show map
    print("[OK] Channel map:")
    for i in range(scan_max):
        ev = await mesh.commands.get_channel(i)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if isinstance(payload, dict):
            print(f"  idx={i} -> {payload.get('channel_name')}")

    gemini_client = genai.Client()
    bot = ChannelGeminiBot(
        mesh=mesh,
        gemini_client=gemini_client,
        channel_idx=chan_idx,
        channel_label=channel_name,
        trigger=trigger,
        model=model,
        max_reply_chars=max_reply_chars,
        history_turns=history_turns,
        dedupe_window_s=dedupe_window_s,
        debug=debug,
    )

    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)

    print(
        f"[OK] Connected: {host}:{port} | listening on {channel_name} (idx={chan_idx}) | trigger='{trigger}'"
    )
    print(f"[TEST] In {channel_name}, send: '{trigger} ping' or 'NAME: {trigger} ping' (expect: @NAME pong)")

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    asyncio.run(main())
