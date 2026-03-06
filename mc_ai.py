#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ listens to channel messages AND direct messages

Features
- Connects over TCP or USB/Serial
- Listens on one MeshCore channel
- Also listens for direct messages (CONTACT_MSG_RECV)
- Triggers on "!ai ..." at start or after "NAME: "
- Channel replies use @[...] prefix
- DM replies do NOT use @[...]
- De-dupes duplicate inbound packets
- Resolves DM destinations via contacts cache (pubkey_prefix -> full public_key)
- Splits long replies safely by UTF-8 BYTES (not chars), accounting for numbering/prefix overhead
  so MeshCore does not truncate messages mid-word.

Env vars (MeshCore transport)
  MESHCORE_TRANSPORT            tcp | serial   (default: tcp)
  MESHCORE_HOST                 required if tcp
  MESHCORE_PORT                 default: 5000
  MESHCORE_SERIAL_PORT          required if serial, e.g. /dev/ttyACM0
  MESHCORE_SERIAL_BAUD          default: 115200
  MESHCORE_CHANNEL_NAME         default: #avl-ai
  CHANNEL_SCAN_MAX              default: 16

Env vars (Bot)
  AI_TRIGGER                    default: !ai
  MAX_REPLY_CHARS               default: 180   # max BYTES of final transmitted message (UTF-8)
  HISTORY_TURNS                 default: 6
  DEBUG                         default: 0
  DEDUPE_WINDOW_S               default: 3.0

Env vars (LLM selection)
  LLM_BACKEND                   gemini | ollama | openai_compat
  SYSTEM_PROMPT                 optional

Gemini
  GEMINI_API_KEY                required if LLM_BACKEND=gemini
  GEMINI_MODEL                  default: gemini-3-flash-preview

Ollama
  OLLAMA_BASE_URL               default: http://127.0.0.1:11434
  OLLAMA_MODEL                  default: llama3.2:latest
  OLLAMA_KEEP_ALIVE             default: 5m

OpenAI-compatible
  LOCAL_LLM_BASE_URL            default: http://127.0.0.1:1234/v1
  LOCAL_LLM_MODEL               default: local-model
  LOCAL_LLM_API_KEY             optional
  LOCAL_LLM_TEMPERATURE         default: 0.3
"""

import asyncio
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx
from meshcore import EventType, MeshCore

try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # noqa: N816


# ---------------------------
# helpers
# ---------------------------

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


def utf8_len(text: str) -> int:
    return len((text or "").encode("utf-8"))


def chunk_text_bytes(text: str, max_bytes: int, prefix: str = "") -> List[str]:
    """
    Split `text` so that (prefix + chunk) fits within `max_bytes` in UTF-8 BYTES.

    This avoids MeshCore truncation when a message exceeds a byte-limit even if
    Python character counts look "safe".

    If a single token is too large, it will be hard-split by bytes.
    """
    text = (text or "").strip()
    prefix = prefix or ""

    if not text:
        return []

    prefix_bytes = utf8_len(prefix)
    if prefix_bytes >= max_bytes:
        raise ValueError("Prefix is too large for max_bytes")

    usable_bytes = max_bytes - prefix_bytes

    if utf8_len(text) <= usable_bytes:
        return [text]

    chunks: List[str] = []
    cur = ""

    # keep whitespace tokens so we can pack nicely
    for tok in re.split(r"(\s+)", text):
        if not tok:
            continue

        candidate = cur + tok
        if utf8_len(candidate) <= usable_bytes:
            cur = candidate
            continue

        if cur.strip():
            chunks.append(cur.strip())

        # token itself might be too large; hard split by bytes
        tok_stripped = tok.strip()
        if tok_stripped and utf8_len(tok_stripped) > usable_bytes:
            piece = ""
            for ch in tok_stripped:
                if utf8_len(piece + ch) <= usable_bytes:
                    piece += ch
                else:
                    if piece:
                        chunks.append(piece)
                    piece = ch
            cur = piece
        else:
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
# LLM clients
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

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        r = await self._http.post(f"{self.base_url}/api/chat", json=payload)
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
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            r = "assistant" if role == "assistant" else "user"
            messages.append({"role": r, "content": msg})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        r = await self._http.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
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
        max_reply_chars: int,  # bytes budget
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

        self._llm_lock = asyncio.Lock()
        self._dedupe_lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}

        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}

    # ---------------- Contacts ----------------

    async def upsert_contact(self, contact: Dict[str, Any]) -> None:
        pk = contact.get("public_key")
        if not isinstance(pk, str) or not pk.strip():
            return

        pubkey = pk.strip().lower()
        prefix = pubkey[:12]

        async with self._contacts_lock:
            self._contacts_by_pubkey[pubkey] = contact
            self._contacts_by_prefix.setdefault(prefix, pubkey)

        if self.debug:
            name = contact.get("name") or contact.get("alias") or ""
            print(f"[DBG] cached contact prefix={prefix} name={name}")

    async def on_contacts_event(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        candidates: List[Dict[str, Any]] = []
        if isinstance(p.get("contacts"), list):
            for c in p["contacts"]:
                if isinstance(c, dict):
                    candidates.append(c)

        if "public_key" in p and isinstance(p.get("public_key"), str):
            candidates.append(p)

        for c in candidates:
            await self.upsert_contact(c)

    async def refresh_contacts_best_effort(self) -> None:
        try:
            if hasattr(self.mesh.commands, "get_contacts"):
                await getattr(self.mesh.commands, "get_contacts")()
                return
            if hasattr(self.mesh.commands, "list_contacts"):
                await getattr(self.mesh.commands, "list_contacts")()
                return
            if self.debug:
                print("[DBG] No get_contacts/list_contacts method found; relying on contact events.")
        except Exception as e:
            if self.debug:
                print(f"[DBG] refresh_contacts_best_effort error: {e}")

    def resolve_dm_dst(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pk = payload.get("public_key")
        if isinstance(pk, str) and pk.strip():
            return {"public_key": pk.strip()}

        prefix = payload.get("pubkey_prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            return None
        prefix = prefix.strip().lower()

        pubkey = self._contacts_by_prefix.get(prefix)
        if not pubkey:
            for pk2 in self._contacts_by_pubkey.keys():
                if pk2.startswith(prefix):
                    pubkey = pk2
                    break

        if not pubkey:
            return None

        return {"public_key": pubkey}

    # ---------------- Helpers ----------------

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
        if not self.trigger_re.search(b):
            return ""
        idx = b.lower().find(self.trigger.lower())
        if idx < 0:
            return ""
        return b[idx + len(self.trigger):].strip(" \t:,-")

    async def dedupe_drop(self, scope: str, ch_idx: int, sender_ts: int, body: str) -> bool:
        key = (scope, ch_idx, sender_ts, body)
        now = time.time()

        async with self._dedupe_lock:
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

    def build_channel_messages(self, sender: str, answer: str) -> List[str]:
        sender_prefix = f"@[{sender}] " if sender else ""

        # 1) conservative first pass to estimate count with a wide numbering prefix
        parts = chunk_text_bytes(
            answer,
            self.max_reply_chars,
            prefix=sender_prefix + "(99/99) ",
        )

        # 2) rebuild with the true numbering width once we know how many parts
        if len(parts) > 1:
            numbering = f"({len(parts)}/{len(parts)}) "
            parts = chunk_text_bytes(
                answer,
                self.max_reply_chars,
                prefix=sender_prefix + numbering,
            )

        out: List[str] = []
        for i, part in enumerate(parts, start=1):
            body = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
            out.append(f"{sender_prefix}{body}" if sender_prefix else body)

        # sanity (debug): ensure we never exceed max bytes
        if self.debug:
            for m in out:
                if utf8_len(m) > self.max_reply_chars:
                    print(f"[DBG] WARN: built channel msg exceeds max bytes: {utf8_len(m)} > {self.max_reply_chars}")

        return out

    def build_dm_messages(self, answer: str) -> List[str]:
        # DM has no @[...] prefix, but may have numbering prefix
        parts = chunk_text_bytes(
            answer,
            self.max_reply_chars,
            prefix="(99/99) ",
        )

        if len(parts) > 1:
            numbering = f"({len(parts)}/{len(parts)}) "
            parts = chunk_text_bytes(
                answer,
                self.max_reply_chars,
                prefix=numbering,
            )

        out: List[str] = []
        for i, part in enumerate(parts, start=1):
            out.append(part if len(parts) == 1 else f"({i}/{len(parts)}) {part}")

        if self.debug:
            for m in out:
                if utf8_len(m) > self.max_reply_chars:
                    print(f"[DBG] WARN: built DM msg exceeds max bytes: {utf8_len(m)} > {self.max_reply_chars}")

        return out

    # ---------------- Event handlers ----------------

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

        if await self.dedupe_drop("chan", self.channel_idx, sender_ts, body):
            return

        if self.debug:
            print(f"[DBG] target-channel msg payload={p}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            if user.lower() == "ping":
                for out in self.build_channel_messages(sender, "pong"):
                    await self.mesh.commands.send_chan_msg(self.channel_idx, out)
                return

            self.history[self.channel_idx].append(("user", user))
            conversation = self.build_conversation(user)

            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[self.channel_idx].append(("assistant", answer))

            for out in self.build_channel_messages(sender, answer):
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

        if await self.dedupe_drop("dm", -1, sender_ts, body):
            return

        if self.debug:
            print(f"[DBG] DM payload={p}")
            print(f"[DBG] DM keys={list(p.keys())}")

        user = self.extract_after_trigger(body)
        if not user:
            return

        dst = self.resolve_dm_dst(p)
        if dst is None:
            await self.refresh_contacts_best_effort()
            dst = self.resolve_dm_dst(p)

        if dst is None:
            if self.debug:
                print("[DBG] Could not resolve DM destination to full public_key; cannot reply.")
            return

        async with self._llm_lock:
            if user.lower() == "ping":
                for out in self.build_dm_messages("pong"):
                    await self.mesh.commands.send_msg(dst, out)
                return

            conversation = self.build_conversation(user)
            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            for out in self.build_dm_messages(answer):
                await self.mesh.commands.send_msg(dst, out)


# ---------------------------
# connection
# ---------------------------

async def create_mesh_connection() -> MeshCore:
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

        if hasattr(MeshCore, "create_serial"):
            return await MeshCore.create_serial(serial_port, baud, auto_reconnect=True)  # type: ignore[attr-defined]

        for alt in ("create_uart", "create_usb", "create_serial_port"):
            if hasattr(MeshCore, alt):
                fn = getattr(MeshCore, alt)
                return await fn(serial_port, baud, auto_reconnect=True)

        raise SystemExit(
            "Your meshcore package does not expose MeshCore.create_serial (or known alternates). "
            "Run: python -c \"from meshcore import MeshCore; print([m for m in dir(MeshCore) if 'create' in m])\""
        )

    raise SystemExit("MESHCORE_TRANSPORT must be one of: tcp | serial")


# ---------------------------
# main
# ---------------------------

async def main() -> None:
    channel_name = env_str("MESHCORE_CHANNEL_NAME", "#avl-ai")
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)  # treated as max UTF-8 bytes budget
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

    mesh = await create_mesh_connection()
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

    mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
    mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)

    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

    await bot.refresh_contacts_best_effort()

    print(f"[OK] Connected | listening on {channel_name} (idx={chan_idx}) | trigger='{trigger}'")
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
