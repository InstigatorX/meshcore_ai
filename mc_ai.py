#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport

- Connects to a MeshCore device over WiFi/TCP OR USB/Serial
- Listens on a specific channel (e.g. #avl-ai)
- Also listens for direct messages (CONTACT_MSG_RECV)
- Triggers on "!ai ..." appearing either at start of message OR after a "NAME: " prefix
- Replies in-channel, prefixed with an @mention of the sender (e.g. "@[iX-HTv4-1] ...")
- Replies in DM WITHOUT @mention prefix
- De-dupes duplicate inbound packets within a short window to avoid double replies

LLM backends supported:
  1) Gemini via google-genai (default): LLM_BACKEND=gemini
  2) Ollama (local):                 LLM_BACKEND=ollama
  3) OpenAI-compatible local server: LLM_BACKEND=openai_compat
     (LM Studio, Open WebUI, vLLM, llama.cpp server in OpenAI mode, etc.)

Env vars (MeshCore transport):
  MESHCORE_TRANSPORT            (default: tcp) one of: tcp | serial
  MESHCORE_HOST                 (required if tcp)
  MESHCORE_PORT                 (default: 5000)
  MESHCORE_SERIAL_PORT          (required if serial) e.g. /dev/ttyACM0
  MESHCORE_SERIAL_BAUD          (default: 115200)
  MESHCORE_CHANNEL_NAME         (default: #avl-ai)  (accepts with/without leading #)
  CHANNEL_SCAN_MAX              (default: 16)

Env vars (Bot):
  AI_TRIGGER                    (default: !ai)  (bash: export AI_TRIGGER='!ai')
  MAX_REPLY_CHARS               (default: 180)
  HISTORY_TURNS                 (default: 6)
  DEBUG                         (default: 0)
  DEDUPE_WINDOW_S               (default: 3.0)

Env vars (LLM selection):
  LLM_BACKEND                   (default: gemini) one of: gemini | ollama | openai_compat
  SYSTEM_PROMPT                 (optional) overrides the system prompt used for all backends

Gemini:
  GEMINI_API_KEY                (required if LLM_BACKEND=gemini)
  GEMINI_MODEL                  (default: gemini-3-flash-preview)

Ollama:
  OLLAMA_BASE_URL               (default: http://127.0.0.1:11434)
  OLLAMA_MODEL                  (default: llama3.2:latest)
  OLLAMA_KEEP_ALIVE             (default: 5m)

OpenAI-compatible:
  LOCAL_LLM_BASE_URL            (default: http://127.0.0.1:1234/v1)
  LOCAL_LLM_MODEL               (default: local-model)
  LOCAL_LLM_API_KEY             (optional)
  LOCAL_LLM_TEMPERATURE         (default: 0.3)
"""

import asyncio
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
from meshcore import MeshCore, EventType

# Gemini is optional; only imported if used
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
    for tok in re.split(r"(\s+)", text):  # keep whitespace tokens
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


# ---------------------------
# LLM clients (Gemini/Ollama/OpenAI-compat)
# ---------------------------

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

        def _call() -> str:
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

    async def aclose(self) -> None:
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
        headers: Dict[str, str] = {}
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

    async def aclose(self) -> None:
        await self._http.aclose()


# ---------------------------
# Bot
# ---------------------------

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

        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            channel_idx: deque(maxlen=history_turns * 2)
        }
        self._lock = asyncio.Lock()

        # (scope, ch_idx, sender_ts, body) -> time
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}

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

    def extract_after_trigger(self, body: str) -> str:
        b = (body or "").strip()
        m = self.trigger_re.search(b)
        if not m:
            return ""
        idx = b.lower().find(self.trigger.lower())
        if idx < 0:
            return ""
        return b[idx + len(self.trigger) :].strip(" \t:,-")

    def _dedupe_drop(self, scope: str, ch_idx: int, sender_ts: int, body: str) -> bool:
        key = (scope, ch_idx, sender_ts, body)
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

    def build_conversation(self, user_text: str) -> List[Tuple[str, str]]:
        hist = list(self.history[self.channel_idx])
        return hist + [("user", user_text)]

    def format_chan_reply(self, sender: str, msg: str) -> str:
        msg = (msg or "").strip()
        if not msg:
            return ""
        return f"@[{sender}] {msg}" if sender else msg

    def format_dm_reply(self, msg: str) -> str:
        # per your requirement: NO @[...] wrapper for DMs
        return (msg or "").strip()

    def extract_dm_peer(self, payload: Dict[str, Any]) -> Optional[Any]:
        """
        MeshCore DM payloads in your environment include:
          - pubkey_prefix
        send_msg() accepts dst: Union[bytes, str, Dict[str, Any]]

        The most reliable thing we have (based on your debug output) is pubkey_prefix,
        so we pass a dict to send_msg: {"pubkey_prefix": "..."}.

        If your MeshCore build later includes 'src' / 'from' / 'peer' fields, you can enhance this.
        """
        pkp = payload.get("pubkey_prefix")
        if isinstance(pkp, str) and pkp.strip():
            return {"pubkey_prefix": pkp.strip()}
        return None

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
        sender_ts = p.get("sender_timestamp")
        if not isinstance(sender_ts, int):
            sender_ts = -1

        if self._dedupe_drop("chan", self.channel_idx, sender_ts, body):
            return

        if self.debug:
            print(f"[DBG] target-channel msg payload={p}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._lock:
            if user.lower() == "ping":
                out = self.format_chan_reply(sender, "pong")
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
                out = self.format_chan_reply(sender, msg)
                if out:
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)

    async def on_dm_msg(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)
        sender_ts = p.get("sender_timestamp")
        if not isinstance(sender_ts, int):
            sender_ts = -1

        # Dedup DMs separately from channel
        if self._dedupe_drop("dm", -1, sender_ts, body):
            return

        if self.debug:
            print(f"[DBG] DM payload={p}")
            print(f"[DBG] DM keys={list(p.keys())}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        peer = self.extract_dm_peer(p)
        if peer is None:
            if self.debug:
                print("[DBG] Could not extract DM peer; cannot reply.")
            return

        async with self._lock:
            if user.lower() == "ping":
                out = self.format_dm_reply("pong")
                if out:
                    await self.mesh.commands.send_msg(peer, out)
                return

            conversation = self.build_conversation(user)
            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            parts = chunk_text(answer, self.max_reply_chars)
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_dm_reply(msg)
                if out:
                    await self.mesh.commands.send_msg(peer, out)


async def create_mesh_connection() -> MeshCore:
    """
    Creates MeshCore connection via:
      - TCP:    MeshCore.create_tcp(host, port, auto_reconnect=True)
      - SERIAL: MeshCore.create_serial(port, baud, auto_reconnect=True)  (method name may vary by version)

    If your meshcore package uses a different constructor name, this will raise a clear error.
    """
    transport = env_str("MESHCORE_TRANSPORT", "tcp").strip().lower()

    if transport == "tcp":
        host = env_str("MESHCORE_HOST", "")
        if not host:
            raise SystemExit("Missing MESHCORE_HOST (required for MESHCORE_TRANSPORT=tcp)")
        port = env_int("MESHCORE_PORT", 5000)
        return await MeshCore.create_tcp(host, port, auto_reconnect=True)

    if transport == "serial":
        serial_port = env_str("MESHCORE_SERIAL_PORT", "")
        if not serial_port:
            raise SystemExit("Missing MESHCORE_SERIAL_PORT (required for MESHCORE_TRANSPORT=serial)")
        baud = env_int("MESHCORE_SERIAL_BAUD", 115200)

        # Most likely constructor name:
        if hasattr(MeshCore, "create_serial"):
            return await MeshCore.create_serial(serial_port, baud, auto_reconnect=True)  # type: ignore[attr-defined]

        # Alternate names some libs use:
        for alt in ("create_uart", "create_usb", "create_serial_port"):
            if hasattr(MeshCore, alt):
                fn = getattr(MeshCore, alt)
                return await fn(serial_port, baud, auto_reconnect=True)

        raise SystemExit(
            "Your meshcore package does not expose MeshCore.create_serial (or known alternates). "
            "Run: python -c \"from meshcore import MeshCore; print([m for m in dir(MeshCore) if 'create' in m])\" "
            "and tell me what constructors you see."
        )

    raise SystemExit("MESHCORE_TRANSPORT must be one of: tcp | serial")


async def main() -> None:
    channel_name = env_str("MESHCORE_CHANNEL_NAME", "#avl-ai")
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")
    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    # LLM backend selection
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

    mesh = await create_mesh_connection()
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
    print(f"[OK] Connected | listening on {channel_name} (idx={chan_idx}) | trigger='{trigger}'")

    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)
    print(f"[OK] Listening for DMs via CONTACT_MSG_RECV (trigger='{trigger}')")

    print(f"[LLM] backend={backend}")
    if backend == "gemini":
        print(f"[LLM] model={env_str('GEMINI_MODEL', 'gemini-3-flash-preview')}")
    elif backend == "ollama":
        print(
            f"[LLM] model={env_str('OLLAMA_MODEL', 'llama3.2:latest')} "
            f"url={env_str('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')}"
        )
    else:
        print(
            f"[LLM] model={env_str('LOCAL_LLM_MODEL', 'local-model')} "
            f"url={env_str('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:1234/v1')}"
        )

    print(f"[TEST] Channel: '{trigger} ping' or 'NAME: {trigger} ping' (expect: @NAME pong)")
    print(f"[TEST] DM: '{trigger} ping' or 'NAME: {trigger} ping' (expect: pong)")

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    asyncio.run(main())
