#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) with Channel + Direct Message support

Key DM fix:
- Some MeshCore builds deliver DMs with only `pubkey_prefix` in the payload.
- We now use `pubkey_prefix` as the DM destination (dict first, then string fallback).

Channel replies: "@[sender] ..."
DM replies: no "@[...]" wrapper
"""

import asyncio
import os
import re
import time
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional, Union, Any

import httpx
from meshcore import MeshCore, EventType

try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # noqa: N816


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
        got_raw = payload.get("channel_name") or payload.get("name") or payload.get("chan_name") or ""
        got = normalize_channel_name(str(got_raw))
        if got == want:
            return idx
    raise RuntimeError(f"Channel '{channel_name}' not found in first {max_channels} channel slots")


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant replying over a low-bandwidth MeshCore channel. "
    "Keep replies short and directly useful (prefer 1–3 sentences). "
    "If uncertain, say so briefly."
)


class LLMClient:
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        if genai is None:
            raise RuntimeError("google-genai not installed; pip install google-genai")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        prompt_lines = [system_prompt, "", "Conversation:"]
        for role, msg in conversation:
            prompt_lines.append(f"{role}: {msg}")
        prompt_lines.append("assistant:")
        prompt = "\n".join(prompt_lines)

        def _call():
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            txt = getattr(resp, "text", None)
            return str(txt).strip() if txt else ""

        txt = await asyncio.to_thread(_call)
        return txt or "I couldn’t generate a response."


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, keep_alive: str = "5m", timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            r = "assistant" if role == "assistant" else "user"
            messages.append({"role": r, "content": msg})

        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive}
        url = f"{self.base_url}/api/chat"
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        return (content or "").strip() or "I couldn’t generate a response."

    async def aclose(self):
        await self._http.aclose()


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            r = "assistant" if role == "assistant" else "user"
            messages.append({"role": r, "content": msg})

        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}

        r = await self._http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return "I couldn’t generate a response."
        content = ((choices[0].get("message") or {}).get("content")) or ""
        return str(content).strip() or "I couldn’t generate a response."

    async def aclose(self):
        await self._http.aclose()


class ChannelLLMBot:
    def __init__(
        self,
        mesh: MeshCore,
        llm: LLMClient,
        channel_idx: int,
        channel_label: str,
        trigger: str,
        max_reply_chars: int,
        history_turns: int,
        dedupe_window_s: float,
        debug: bool,
        system_prompt: str,
    ):
        self.mesh = mesh
        self.llm = llm
        self.channel_idx = channel_idx
        self.channel_label = channel_label
        self.trigger = trigger
        self.max_reply_chars = max_reply_chars
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s
        self.system_prompt = system_prompt

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)

        self.history: Dict[int, Deque[Tuple[str, str]]] = {channel_idx: deque(maxlen=history_turns * 2)}
        self._lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}  # (scope, ch, sender_ts, body) -> time

    @staticmethod
    def split_sender_and_body(text: str) -> Tuple[str, str]:
        t = (text or "").strip()
        if ": " in t:
            name, body = t.split(": ", 1)
            name = name.strip()
            body = body.strip()
            if name and len(name) <= 40:
                return name, body
        return "", t

    @staticmethod
    def format_channel_reply(sender: str, msg: str) -> str:
        msg = (msg or "").strip()
        if not msg:
            return ""
        return f"@[{sender}] {msg}" if sender else msg

    @staticmethod
    def format_dm_reply(msg: str) -> str:
        return (msg or "").strip()

    def extract_after_trigger(self, body: str) -> str:
        b = (body or "").strip()
        if not self.trigger_re.search(b):
            return ""
        idx = b.lower().find(self.trigger.lower())
        if idx < 0:
            return ""
        return b[idx + len(self.trigger):].strip(" \t:,-")

    def _dedupe_key(self, scope: str, payload: dict, body: str) -> Tuple[str, int, int, str]:
        ch = payload.get("channel_idx")
        if not isinstance(ch, int):
            ch = -1
        st = payload.get("sender_timestamp")
        if not isinstance(st, int):
            st = payload.get("timestamp")
        if not isinstance(st, int):
            st = 0
        return (scope, ch, st, body)

    def _dedupe_drop(self, scope: str, payload: dict, body: str) -> bool:
        key = self._dedupe_key(scope, payload, body)
        now = time.time()

        for k, t0 in list(self._seen_ts.items()):
            if now - t0 > self.dedupe_window_s:
                self._seen_ts.pop(k, None)

        if key in self._seen_ts:
            if self.debug:
                print(f"[DBG] duplicate dropped key={key}")
            return True

        self._seen_ts[key] = now
        return False

    def build_conversation(self, user_text: str) -> List[Tuple[str, str]]:
        hist = list(self.history[self.channel_idx])
        return hist + [("user", user_text)]

    def _extract_dm_peer(self, payload: dict) -> Optional[Union[str, bytes, dict]]:
        """
        Your DM payload example includes ONLY:
          pubkey_prefix='96cfa27f6f9c'
        So we use that as the destination.
        """
        # Most important for your case:
        pfx = payload.get("pubkey_prefix")
        if isinstance(pfx, str) and pfx:
            # Prefer dict form first (explicit)
            return {"pubkey_prefix": pfx}

        # Other common possibilities (kept as fallback)
        for k in ("contact_uri", "peer_uri", "from_uri", "uri", "src", "dst"):
            v = payload.get(k)
            if isinstance(v, (str, bytes)) and v:
                return v
        for k in ("contact", "peer", "from", "sender"):
            v = payload.get(k)
            if isinstance(v, dict) and v:
                return v
        return None

    async def _send_dm(self, peer: Union[str, bytes, dict], text: str) -> None:
        msg = (text or "").strip()
        if not msg:
            return

        # Try dict form first, then fallback to raw pubkey_prefix string if needed.
        candidates: List[Union[str, bytes, dict]] = [peer]
        if isinstance(peer, dict) and "pubkey_prefix" in peer:
            candidates.append(str(peer["pubkey_prefix"]))

        last_err: Optional[Exception] = None

        for dst in candidates:
            try:
                send_with_retry = getattr(self.mesh.commands, "send_msg_with_retry", None)
                if callable(send_with_retry):
                    ev = await send_with_retry(dst, msg)
                else:
                    send_msg = getattr(self.mesh.commands, "send_msg", None)
                    if not callable(send_msg):
                        raise RuntimeError("mesh.commands missing send_msg/send_msg_with_retry")
                    ev = await send_msg(dst, msg)

                if self.debug:
                    print(f"[DBG] DM sent dst={dst!r} ev.type={getattr(ev,'type',None)} ev.payload={getattr(ev,'payload',None)}")
                return
            except Exception as e:
                last_err = e
                if self.debug:
                    print(f"[DBG] DM send failed dst={dst!r} err={e}")

        raise RuntimeError(f"All DM send attempts failed. Last error: {last_err}")

    async def on_channel_msg(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return
        if p.get("channel_idx") != self.channel_idx:
            return
        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)

        if self._dedupe_drop("chan", p, body):
            return

        if self.debug:
            print(f"[DBG] CHAN payload={p}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._lock:
            if user.lower() == "ping":
                out = self.format_channel_reply(sender, "pong")
                if out:
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)
                return

            self.history[self.channel_idx].append(("user", user))
            conversation = self.build_conversation(user)

            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[self.channel_idx].append(("assistant", answer))

            parts = chunk_text(answer, self.max_reply_chars)
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_channel_reply(sender, msg)
                if out:
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)

    async def on_direct_msg(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        if self.debug:
            print(f"[DBG] DM payload={p}")
            print(f"[DBG] DM keys={sorted(list(p.keys()))}")

        sender, body = self.split_sender_and_body(text)

        if self._dedupe_drop("dm", p, body):
            return

        user = self.extract_after_trigger(body)
        if not user:
            return

        peer = self._extract_dm_peer(p)
        if peer is None:
            if self.debug:
                print("[DBG] Could not extract DM peer; cannot reply.")
            return

        async with self._lock:
            if user.lower() == "ping":
                await self._send_dm(peer, self.format_dm_reply("pong"))
                return

            self.history[self.channel_idx].append(("user", user))
            conversation = self.build_conversation(user)

            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[self.channel_idx].append(("assistant", answer))

            parts = chunk_text(answer, self.max_reply_chars)
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                await self._send_dm(peer, self.format_dm_reply(msg))


async def main() -> None:
    host = env_str("MESHCORE_HOST", "")
    if not host:
        raise SystemExit("Missing MESHCORE_HOST")

    port = env_int("MESHCORE_PORT", 5000)
    channel_name = env_str("MESHCORE_CHANNEL_NAME", "#avl-ai")
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")

    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    backend = env_str("LLM_BACKEND", "gemini").lower()

    llm: LLMClient
    if backend == "gemini":
        api_key = env_str("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("Missing GEMINI_API_KEY (required for LLM_BACKEND=gemini)")
        model = env_str("GEMINI_MODEL", "gemini-3-flash-preview")
        llm = GeminiClient(api_key=api_key, model=model)

    elif backend == "ollama":
        base_url = env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = env_str("OLLAMA_MODEL", "llama3.2:latest")
        keep_alive = env_str("OLLAMA_KEEP_ALIVE", "5m")
        llm = OllamaClient(base_url=base_url, model=model, keep_alive=keep_alive)

    elif backend == "openai_compat":
        base_url = env_str("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
        model = env_str("LOCAL_LLM_MODEL", "local-model")
        api_key = env_str("LOCAL_LLM_API_KEY", "") or None
        temperature = env_float("LOCAL_LLM_TEMPERATURE", 0.3)
        llm = OpenAICompatClient(base_url=base_url, model=model, api_key=api_key, temperature=temperature)

    else:
        raise SystemExit("LLM_BACKEND must be one of: gemini | ollama | openai_compat")

    mesh = await MeshCore.create_tcp(host, port, auto_reconnect=True)
    await mesh.start_auto_message_fetching()

    chan_idx = await resolve_channel_idx(mesh, channel_name, max_channels=scan_max)

    print("[OK] Channel map:")
    for i in range(scan_max):
        ev = await mesh.commands.get_channel(i)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if isinstance(payload, dict):
            print(f"  idx={i} -> {payload.get('channel_name')}")

    bot = ChannelLLMBot(
        mesh=mesh,
        llm=llm,
        channel_idx=chan_idx,
        channel_label=channel_name,
        trigger=trigger,
        max_reply_chars=max_reply_chars,
        history_turns=history_turns,
        dedupe_window_s=dedupe_window_s,
        debug=debug,
        system_prompt=system_prompt,
    )

    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_direct_msg)

    print(f"[OK] Connected: {host}:{port} | listening on {channel_name} (idx={chan_idx}) | trigger='{trigger}'")
    print(f"[OK] Listening for DMs via CONTACT_MSG_RECV (trigger='{trigger}')")
    print(f"[LLM] backend={backend}")

    print(f"[TEST] Channel: '{trigger} ping' or 'NAME: {trigger} ping' (expect: @NAME pong)")
    print(f"[TEST] DM: '{trigger} ping' or 'NAME: {trigger} ping' (expect: pong)")

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    asyncio.run(main())
