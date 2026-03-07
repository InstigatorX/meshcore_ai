#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ listens to channel messages AND direct messages

Features
- Connects over TCP or USB/Serial
- Owns reconnect logic itself (MeshCore internal auto_reconnect is disabled)
- Listens on MULTIPLE MeshCore channels defined in config.
- Also listens for direct messages (CONTACT_MSG_RECV)
- Triggers on "!ai ..." at start or after "NAME: "
- Channel replies use @[...] prefix
- DM replies do NOT use @[...]
- De-dupes duplicate inbound packets
- Adds in-flight dedupe to prevent duplicate LLM calls
- Resolves DM destinations via contacts cache (pubkey_prefix -> full public_key)
- Splits long replies safely, accounting for numbering/prefix overhead
- Handles concurrent LLM requests for better responsiveness.
- Persistent LLM client across mesh reconnections.
- "ping" command now works WITHOUT the trigger (just type "ping").
- Maintains conversation history per-user (based on pubkey_prefix).
- **FIXED:** Robust reconnect logic with guaranteed cleanup and health monitoring.
"""

import asyncio
import logging
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
from meshcore import EventType, MeshCore

# Check for google-genai library
try:
    import google.genai as genai
    # Suppress noisy INFO logs from google_genai library (e.g., "AFC is enabled...")
    logging.getLogger("google_genai").setLevel(logging.WARNING)
except ImportError:
    genai = None


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


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def normalize_channel_name(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("#"):
        n = n[1:]
    return n.strip().lower()


def _split_with_budget(text: str, budget: int) -> List[str]:
    """
    Splits text into chunks that fit within the budget.
    """
    text = (text or "").strip()
    if not text:
        return []
    if budget <= 0:
        raise ValueError("budget must be > 0")

    words = text.split(' ')
    chunks: List[str] = []
    current_chunk = ""

    for word in words:
        if not word:
            continue
        
        if len(word) > budget:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            for i in range(0, len(word), budget):
                chunks.append(word[i:i+budget])
        elif len(current_chunk) + len(word) + (1 if current_chunk else 0) > budget:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_text(text: str, max_len: int, prefix_len: int = 0) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    budget = max_len - prefix_len
    if budget <= 0:
        raise ValueError("max_len must be greater than prefix_len")

    return _split_with_budget(text, budget)


def _build_number_prefix(i: int, n: int) -> str:
    return f"({i}/{n}) " if n > 1 else ""


def split_for_transport(
    text: str,
    max_len: int,
    fixed_prefix: str = "",
    number_parts: bool = True,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    fixed_len = len(fixed_prefix)

    if fixed_len + len(text) <= max_len:
        return [f"{fixed_prefix}{text}"]

    if not number_parts:
        parts = chunk_text(text, max_len=max_len, prefix_len=fixed_len)
        return [f"{fixed_prefix}{p}" for p in parts]

    n_guess = max(2, (fixed_len + len(text) + max_len - 1) // max_len)
    n_guess = min(max(n_guess, 2), 999)

    while True:
        num_len = len(_build_number_prefix(n_guess, n_guess))
        parts = chunk_text(text, max_len=max_len, prefix_len=fixed_len + num_len)
        n = len(parts)
        if n == n_guess:
            break
        n_guess = n

    out: List[str] = []
    for i, part in enumerate(parts, start=1):
        np = _build_number_prefix(i, n)
        msg = f"{fixed_prefix}{np}{part}"
        if len(msg) > max_len:
            msg = msg[:max_len]
        out.append(msg)
    return out


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant replying over a low-bandwidth MeshCore channel. "
    "Keep replies short and directly useful (prefer 1–3 sentences). "
)

DEFAULT_PING_TEMPLATE = "🤖 Ack {who}\n[{stats}]"

# ---------------------------
# LLM clients
# ---------------------------

class LLMClient:
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        raise NotImplementedError
    async def aclose(self) -> None:
        return None

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        if genai is None:
            raise RuntimeError("google-genai not installed; pip install google-genai")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        prompt_parts = [system_prompt, "\nConversation:"]
        for role, msg in conversation:
            prompt_parts.append(f"{role}: {msg}")
        prompt_parts.append("assistant:")
        prompt = "\n".join(prompt_parts)

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
        r = await self._http.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        return (content or "").strip() or "I couldn’t generate a response."

    async def aclose(self) -> None:
        await self._http.aclose()

class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, temperature: float = 0.3, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            r = "assistant" if role == "assistant" else "user"
            messages.append({"role": r, "content": msg})
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        r = await self._http.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices: return "I couldn’t generate a response."
        content = ((choices[0].get("message") or {}).get("content")) or ""
        return str(content).strip() or "I couldn’t generate a response."

    async def aclose(self) -> None:
        await self._http.aclose()

# ---------------------------
# Bot
# ---------------------------

class ChannelLLMBot:
    def __init__(self, mesh: MeshCore, llm: LLMClient, monitored_channels: Dict[int, str], trigger: str, max_reply_chars: int, history_turns: int, dedupe_window_s: float, debug: bool, system_prompt: str, include_requester_context: bool, ping_reply_template: str, generation: int, current_generation_ref):
        self.mesh = mesh
        self.llm = llm
        self.monitored_channels = monitored_channels
        self.trigger = trigger
        self.max_reply_chars = max_reply_chars
        self.history_turns = history_turns
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s
        self.system_prompt = system_prompt
        self.include_requester_context = include_requester_context
        self.ping_reply_template = ping_reply_template
        self.generation = generation
        self.current_generation_ref = current_generation_ref
        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)
        self.history: Dict[str, Deque[Tuple[str, str]]] = {}
        self._history_lock = asyncio.Lock()
        self._dedupe_lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}
        self._inflight: Dict[Tuple[str, int, int, str], float] = {}
        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}

    def is_stale(self) -> bool:
        return self.generation != self.current_generation_ref()

    async def _safe_send_dm(self, dst: Dict[str, Any], text: str):
        """Attempts multiple MeshCore methods to send a DM."""
        for method_name in ["send_msg", "send_direct_msg", "send_priv_msg"]:
            if hasattr(self.mesh.commands, method_name):
                try:
                    method = getattr(self.mesh.commands, method_name)
                    await method(dst, text)
                    if self.debug: print(f"[DBG] DM sent via {method_name}")
                    return
                except Exception as e:
                    if self.debug: print(f"[DBG] DM {method_name} failed: {e}")
        print("[ERROR] Failed to find a working DM send method in meshcore library")

    async def upsert_contact(self, contact: Dict[str, Any]) -> None:
        pk = contact.get("public_key")
        if not isinstance(pk, str) or not pk.strip(): return
        pubkey = pk.strip().lower()
        prefix = pubkey[:12]
        async with self._contacts_lock:
            self._contacts_by_pubkey[pubkey] = contact
            self._contacts_by_prefix.setdefault(prefix, pubkey)

    async def on_contacts_event(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return
        candidates = []
        if isinstance(p.get("contacts"), list):
            for c in p["contacts"]:
                if isinstance(c, dict): candidates.append(c)
        if "public_key" in p and isinstance(p.get("public_key"), str): candidates.append(p)
        for c in candidates: await self.upsert_contact(c)

    async def refresh_contacts_best_effort(self) -> None:
        try:
            for cmd in ["get_contacts", "list_contacts"]:
                if hasattr(self.mesh.commands, cmd):
                    await getattr(self.mesh.commands, cmd)()
                    return
        except Exception as e:
            if self.debug: print(f"[DBG] refresh_contacts error: {e}")

    async def resolve_dm_dst(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Determines the destination for a DM reply.
        Ensures 'public_key' field exists for library compatibility.
        """
        # 1. Full public_key
        pk = payload.get("public_key")
        if isinstance(pk, str) and pk.strip():
            return {"public_key": pk.strip()}

        # 2. Resolve prefix from cache
        prefix = payload.get("pubkey_prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            return None
        
        prefix = prefix.strip().lower()
        async with self._contacts_lock:
            pubkey = self._contacts_by_prefix.get(prefix)
            if not pubkey:
                for pk2 in self._contacts_by_pubkey.keys():
                    if pk2.startswith(prefix):
                        pubkey = pk2
                        break
        
        if pubkey:
            return {"public_key": pubkey}

        # 3. Fallback: Use prefix in the public_key field
        if self.debug:
            print(f"[DBG] DM destination resolved as prefix fallback: {prefix}")
        return {"public_key": prefix}

    def build_ping_reply(self, payload: Dict[str, Any], sender_name_from_text: str = "") -> str:
        who = sender_name_from_text
        if not who:
            pk = payload.get("pubkey_prefix")
            who = pk[:8] if isinstance(pk, str) and pk else "unknown"
        stats_parts = []
        snr = payload.get("SNR")
        if isinstance(snr, (int, float)):
            q = "Poor"
            if snr > 10: q = "Ok"
            if snr > 20: q = "Good"
            if snr > 30: q = "Great"
            stats_parts.append(f"SNR: {snr:.0f}dB ({q})")
        path_len = payload.get("path_len")
        if isinstance(path_len, int): stats_parts.append(f"Hops: {max(0, path_len - 1)}")
        stats_str = " | ".join(stats_parts)
        try:
            reply = self.ping_reply_template.format(who=who, stats=stats_str)
        except:
            reply = DEFAULT_PING_TEMPLATE.format(who=who, stats=stats_str)
        return reply.replace("[]", "").strip()

    def build_requester_context(self, scope: str, payload: Dict[str, Any], channel_idx: int = -1, sender_name: str = "") -> str:
        if not self.include_requester_context: return ""
        lines = [f"Requester context:\n- scope: {scope}"]
        if scope == "chan" and channel_idx != -1:
            lines.append(f"- channel: {self.monitored_channels.get(channel_idx, 'unknown')}")
        if sender_name: lines.append(f"- sender_name: {sender_name}")
        for f in ["pubkey_prefix", "SNR", "path_len", "sender_timestamp"]:
            v = payload.get(f)
            if v is not None: lines.append(f"- {f}: {v}")
        return "\n".join(lines)

    async def get_conversation_snapshot(self, user_id: str) -> List[Tuple[str, str]]:
        async with self._history_lock:
            if user_id not in self.history: self.history[user_id] = deque(maxlen=self.history_turns * 2)
            return list(self.history[user_id])

    async def append_to_history(self, user_id: str, user_msg: str, assistant_msg: str) -> None:
        async with self._history_lock:
            if user_id in self.history:
                self.history[user_id].append(("user", user_msg))
                self.history[user_id].append(("assistant", assistant_msg))

    def split_sender_and_body(self, text: str) -> Tuple[str, str]:
        t = (text or "").strip()
        if ": " in t:
            n, b = t.split(": ", 1)
            if n and len(n) <= 40: return n.strip(), b.strip()
        return "", t

    def extract_after_trigger(self, body: str) -> str:
        b = (body or "").strip()
        if not self.trigger_re.search(b): return ""
        idx = b.lower().find(self.trigger.lower())
        return b[idx + len(self.trigger):].strip(" \t:,-")

    async def dedupe_enter(self, scope: str, ch_idx: int, sender_ts: int, body: str) -> Optional[Tuple[str, int, int, str]]:
        key = (scope, ch_idx, sender_ts, body)
        now = time.time()
        async with self._dedupe_lock:
            self._seen_ts = {k: v for k, v in self._seen_ts.items() if now - v < self.dedupe_window_s}
            self._inflight = {k: v for k, v in self._inflight.items() if now - v < 30.0}
            if key in self._seen_ts or key in self._inflight: return None
            self._inflight[key] = now
            return key

    async def dedupe_exit(self, key: Tuple[str, int, int, str]) -> None:
        async with self._dedupe_lock:
            self._inflight.pop(key, None)
            self._seen_ts[key] = time.time()

    async def on_channel_msg(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return
        chan_idx = p.get("channel_idx")
        if chan_idx not in self.monitored_channels: return
        sender, body = self.split_sender_and_body(p.get("text", ""))
        dedupe_key = await self.dedupe_enter("chan", chan_idx, p.get("sender_timestamp", -1), body)
        if not dedupe_key: return
        try:
            if body.strip().lower() == "ping":
                for out in split_for_transport(self.build_ping_reply(p, sender), self.max_reply_chars, f"@[{sender}] "):
                    await self.mesh.commands.send_chan_msg(chan_idx, out)
                return
            user_msg = self.extract_after_trigger(body)
            if not user_msg: return
            user_id = p.get("pubkey_prefix") or sender
            hist = await self.get_conversation_snapshot(user_id)
            sys_prompt = f"{self.system_prompt}\n\n{self.build_requester_context('chan', p, chan_idx, sender)}"
            try: answer = await self.llm.generate(sys_prompt, hist + [("user", user_msg)])
            except Exception as e: answer = f"LLM error: {e}"
            await self.append_to_history(user_id, user_msg, answer)
            for out in split_for_transport(answer, self.max_reply_chars, f"@[{sender}] "):
                await self.mesh.commands.send_chan_msg(chan_idx, out)
        finally: await self.dedupe_exit(dedupe_key)

    async def on_dm_msg(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return

        # DMs are raw text, don't use split_sender_and_body
        raw_text = p.get("text") or p.get("body") or ""
        if not raw_text: return
        
        body = raw_text.strip()
        sender_ts = p.get("sender_timestamp", -1)

        dedupe_key = await self.dedupe_enter("dm", -1, sender_ts, body)
        if not dedupe_key: return

        try:
            if self.debug:
                print(f"[DBG] DM Processing: '{body}' from prefix {p.get('pubkey_prefix')}")

            dst = await self.resolve_dm_dst(p)
            if not dst:
                if self.debug: print("[DBG] DM ignored: Could not resolve destination")
                return

            # Check for ping
            if body.lower() == "ping":
                reply_text = self.build_ping_reply(p)
                for out in split_for_transport(reply_text, self.max_reply_chars):
                    await self._safe_send_dm(dst, out)
                return

            # Standard LLM logic (DMs do NOT require trigger)
            user_msg = body
            user_id = p.get("pubkey_prefix") or p.get("public_key") or "unknown_dm"
            
            hist = await self.get_conversation_snapshot(user_id)
            sys_prompt = f"{self.system_prompt}\n\n{self.build_requester_context('dm', p)}"
            
            if self.debug: print(f"[DBG] DM calling LLM for user {user_id}")
            
            try:
                answer = await self.llm.generate(sys_prompt, hist + [("user", user_msg)])
            except Exception as e:
                answer = f"LLM error: {e}"

            await self.append_to_history(user_id, user_msg, answer)

            for out in split_for_transport(answer, self.max_reply_chars):
                await self._safe_send_dm(dst, out)

        except Exception as e:
            print(f"[ERROR] on_dm_msg crashed: {e}")
        finally:
            await self.dedupe_exit(dedupe_key)

# ---------------------------
# execution
# ---------------------------

async def create_mesh_connection() -> MeshCore:
    transport = env_str("MESHCORE_TRANSPORT", "tcp").lower()
    if transport == "tcp":
        return await MeshCore.create_tcp(env_str("MESHCORE_HOST", ""), env_int("MESHCORE_PORT", 5000), auto_reconnect=False)
    # Serial fallback
    port = env_str("MESHCORE_SERIAL_PORT", "")
    baud = env_int("MESHCORE_SERIAL_BAUD", 115200)
    for m in ["create_serial", "create_uart", "create_usb"]:
        if hasattr(MeshCore, m): return await getattr(MeshCore, m)(port, baud, auto_reconnect=False)
    raise RuntimeError("No serial transport found in MeshCore")

_CURRENT_GENERATION = 0

async def run_bot_once(generation: int, llm: LLMClient) -> None:
    mesh = await create_mesh_connection()
    if not mesh: raise RuntimeError("MeshCore connection failed")
    
    try:
        await asyncio.sleep(1.0)
        await mesh.start_auto_message_fetching()
        
        want_names = {normalize_channel_name(n) for n in env_str("MESHCORE_CHANNELS", "#avl-ai").split(",") if n.strip()}
        monitored = {}
        for i in range(env_int("CHANNEL_SCAN_MAX", 16)):
            ev = await mesh.commands.get_channel(i)
            if ev.type == EventType.ERROR: continue
            name = str(ev.payload.get("channel_name") or ev.payload.get("name") or "")
            if normalize_channel_name(name) in want_names: monitored[i] = name

        if not monitored: raise RuntimeError(f"Requested channels {want_names} not found")

        bot = ChannelLLMBot(mesh, llm, monitored, env_str("AI_TRIGGER", "!ai"), env_int("MAX_REPLY_CHARS", 140), env_int("HISTORY_TURNS", 6), env_float("DEDUPE_WINDOW_S", 3.0), env_bool("DEBUG", False), env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT), env_bool("INCLUDE_REQUESTER_CONTEXT", True), env_str("PING_REPLY_TEMPLATE", DEFAULT_PING_TEMPLATE), generation, lambda: _CURRENT_GENERATION)
        
        mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
        mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
        mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)
        await bot.refresh_contacts_best_effort()

        print(f"[OK] Gen {generation} connected to: {list(monitored.values())}")
        
        loop = asyncio.get_running_loop()
        exit_event = loop.create_future()
        mesh.subscribe(EventType.DISCONNECTED, lambda _: not exit_event.done() and exit_event.set_result(None))

        async def health_check():
            while not exit_event.done():
                try:
                    await asyncio.wait_for(mesh.commands.get_channel(next(iter(monitored))), timeout=5.0)
                    await asyncio.sleep(15)
                except:
                    if not exit_event.done(): exit_event.set_result(None); break

        hc_task = asyncio.create_task(health_check())
        await exit_event
        hc_task.cancel()
    finally:
        await mesh.aclose()

async def main():
    global _CURRENT_GENERATION
    backend = env_str("LLM_BACKEND", "gemini").lower()
    if backend == "gemini": llm = GeminiClient(env_str("GEMINI_API_KEY", ""), env_str("GEMINI_MODEL", "gemini-3-flash-preview"))
    elif backend == "ollama": llm = OllamaClient(env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), env_str("OLLAMA_MODEL", "llama3.2:latest"))
    else: llm = OpenAICompatClient(env_str("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"), env_str("LOCAL_LLM_MODEL", "local-model"))

    base_delay = env_int("RECONNECT_DELAY_S", 5)
    delay = base_delay
    
    while True:
        try:
            _CURRENT_GENERATION += 1
            await run_bot_once(_CURRENT_GENERATION, llm)
            delay = base_delay
        except Exception as e:
            print(f"[WARN] Session failed: {e}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, env_int("RECONNECT_MAX_DELAY_S", 60))

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
