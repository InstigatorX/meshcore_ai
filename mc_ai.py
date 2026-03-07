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
- "ping" command replies with configurable requester context.

Env vars (MeshCore transport)
  MESHCORE_TRANSPORT            tcp | serial   (default: tcp)
  MESHCORE_HOST                 required if tcp
  MESHCORE_PORT                 default: 5000
  MESHCORE_SERIAL_PORT          required if serial, e.g. /dev/ttyACM0
  MESHCORE_SERIAL_BAUD          default: 115200
  MESHCORE_CHANNELS             Comma-separated list of channels to listen on.
                                default: #avl-ai
  CHANNEL_SCAN_MAX              default: 16

Env vars (Bot)
  AI_TRIGGER                    default: !ai   (bash: export AI_TRIGGER='!ai')
  MAX_REPLY_CHARS               default: 140   # final transmitted message max (safer default)
  HISTORY_TURNS                 default: 6
  DEBUG                         default: 0
  DEDUPE_WINDOW_S               default: 3.0
  INCLUDE_REQUESTER_CONTEXT     default: 1
  PING_REPLY_TEMPLATE           default: "🤖 Ack {who}\n[{stats}]"
  RECONNECT_DELAY_S             default: 5
  RECONNECT_MAX_DELAY_S         default: 60

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
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
from meshcore import EventType, MeshCore

# Check for google-genai library
try:
    import google.genai as genai
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
    Handles long words by hard-splitting them if necessary.
    Correctly accounts for spaces between words.
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
        
        # If a single word is longer than the budget, it must be split
        if len(word) > budget:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Hard split the long word
            for i in range(0, len(word), budget):
                chunks.append(word[i:i+budget])
        # If adding the next word exceeds the budget, push current chunk
        elif len(current_chunk) + len(word) + (1 if current_chunk else 0) > budget:
            chunks.append(current_chunk)
            current_chunk = word
        # Otherwise, add the word to the current chunk
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
    """
    Build final transmitted messages that will fit after adding:
      fixed_prefix + optional '(i/n) ' prefix
    """
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
            # This should theoretically not happen with correct splitting logic,
            # but as a safety measure, truncate.
            msg = msg[:max_len]
        out.append(msg)
    return out


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant replying over a low-bandwidth MeshCore channel. "
    "Keep replies short and directly useful (prefer 1–3 sentences). "
    "If uncertain, say so briefly."
)

# Default template for ping replies if env var is not set.
DEFAULT_PING_TEMPLATE = "🤖 Ack {who}\n[{stats}]"

# Index used for DM history storage
DM_HISTORY_IDX = -1

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
        # Construct prompt more efficiently
        prompt_parts = [system_prompt, "\nConversation:"]
        for role, msg in conversation:
            prompt_parts.append(f"{role}: {msg}")
        prompt_parts.append("assistant:")
        prompt = "\n".join(prompt_parts)

        def _call() -> str:
            # This is a blocking call, so we run it in a separate thread
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
            # Map 'assistant' and 'user' to standard roles
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
        monitored_channels: Dict[int, str],
        trigger: str,
        max_reply_chars: int,
        history_turns: int,
        dedupe_window_s: float,
        debug: bool,
        system_prompt: str,
        include_requester_context: bool,
        ping_reply_template: str,
        generation: int,
        current_generation_ref,
    ):
        self.mesh = mesh
        self.llm = llm
        self.monitored_channels = monitored_channels
        self.trigger = trigger
        self.max_reply_chars = max_reply_chars
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s
        self.system_prompt = system_prompt
        self.include_requester_context = include_requester_context
        self.ping_reply_template = ping_reply_template
        self.generation = generation
        self.current_generation_ref = current_generation_ref

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)

        # History is shared state accessed by concurrent message handlers.
        # Keyed by channel index. Index -1 is used for DMs.
        self.history: Dict[int, Deque[Tuple[str, str]]] = {}
        
        # Initialize history buffers for all monitored channels
        for idx in self.monitored_channels:
             self.history[idx] = deque(maxlen=history_turns * 2)
        # Initialize history buffer for DMs
        self.history[DM_HISTORY_IDX] = deque(maxlen=history_turns * 2)

        # Lock to protect access to self.history
        self._history_lock = asyncio.Lock()

        self._dedupe_lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}
        self._inflight: Dict[Tuple[str, int, int, str], float] = {}

        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}

    # ---------------- lifecycle ----------------

    def is_stale(self) -> bool:
        return self.generation != self.current_generation_ref()

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
        if self.is_stale():
            return

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
            # Try different methods depending on meshcore version
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

    async def resolve_dm_dst(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pk = payload.get("public_key")
        if isinstance(pk, str) and pk.strip():
            return {"public_key": pk.strip()}

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

        if not pubkey:
            return None
        return {"public_key": pubkey}

    # ---------------- Requester context ----------------

    def build_ping_reply(self, payload: Dict[str, Any], sender_name_from_text: str = "") -> str:
        """Builds a human-readable reply with network stats for 'ping' requests using a configurable template."""
        # 1. Identify whomever we are replying to
        who = sender_name_from_text
        if not who:
            pk = payload.get("pubkey_prefix")
            if isinstance(pk, str) and pk:
                who = pk[:8]  # Use first 8 chars of prefix if name unavailable
            else:
                who = "unknown"

        # 2. Gather network statistics
        stats_parts = []

        # SNR (Signal-to-Noise Ratio)
        snr = payload.get("SNR")
        if isinstance(snr, (int, float)):
            # Provide a basic qualitative interpretation along with the raw value
            quality = "Poor"
            if snr > 10: quality = "Ok"
            if snr > 20: quality = "Good"
            if snr > 30: quality = "Great"
            stats_parts.append(f"SNR: {snr:.0f}dB ({quality})")

        # Hops (Path Length)
        path_len = payload.get("path_len")
        if isinstance(path_len, int):
            # path_len is usually total nodes in path, including start.
            # Hops = path_len - 1. (1 means direct).
            hops = max(0, path_len - 1)
            stats_parts.append(f"Hops: {hops}")

        stats_str = " | ".join(stats_parts)

        # 3. Assemble final string using template
        try:
            reply = self.ping_reply_template.format(who=who, stats=stats_str)
        except Exception as e:
            print(f"[WARN] Error formatting PING_REPLY_TEMPLATE: {e}. Falling back to default.")
            reply = DEFAULT_PING_TEMPLATE.format(who=who, stats=stats_str)

        # Cleanup empty brackets if stats were missing but template had brackets.
        # This happens if template is like "[{stats}]" and stats_str is empty.
        reply = reply.replace("[]", "").strip()

        return reply

    def build_requester_context(
        self,
        scope: str,
        payload: Dict[str, Any],
        channel_idx: int = -1,
        sender_name: str = "",
    ) -> str:
        if not self.include_requester_context:
            return ""

        lines: List[str] = ["Requester context:"]
        lines.append(f"- scope: {'direct_message' if scope == 'dm' else 'channel_message'}")

        if scope == "chan" and channel_idx != -1:
            chan_name = self.monitored_channels.get(channel_idx, "unknown")
            lines.append(f"- channel_name: {chan_name}")
            lines.append(f"- channel_idx: {channel_idx}")

        if sender_name:
            lines.append(f"- sender_name: {sender_name}")

        pubkey_prefix = payload.get("pubkey_prefix")
        if isinstance(pubkey_prefix, str) and pubkey_prefix:
            lines.append(f"- sender_pubkey_prefix: {pubkey_prefix}")

        for field in ["SNR", "path_len", "sender_timestamp", "txt_type"]:
            val = payload.get(field)
            if isinstance(val, (int, float)):
                 lines.append(f"- {field.lower()}: {val}")

        return "\n".join(lines)

    def effective_system_prompt(self, requester_context: str) -> str:
        if requester_context:
            return f"{self.system_prompt}\n\n{requester_context}"
        return self.system_prompt

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

    async def dedupe_enter(self, scope: str, ch_idx: int, sender_ts: int, body: str) -> Optional[Tuple[str, int, int, str]]:
        key = (scope, ch_idx, sender_ts, body)
        now = time.time()

        async with self._dedupe_lock:
            # Lazy cleanup of old entries
            for k, t0 in list(self._seen_ts.items()):
                if now - t0 > self.dedupe_window_s:
                    self._seen_ts.pop(k, None)

            for k, t0 in list(self._inflight.items()):
                if now - t0 > max(self.dedupe_window_s, 30.0): # Longer timeout for inflight
                    self._inflight.pop(k, None)

            if key in self._seen_ts:
                if self.debug:
                    print(f"[DBG] duplicate dropped (seen) key={key}")
                return None

            if key in self._inflight:
                if self.debug:
                    print(f"[DBG] duplicate dropped (inflight) key={key}")
                return None

            self._inflight[key] = now
            return key

    async def dedupe_exit(self, key: Tuple[str, int, int, str]) -> None:
        async with self._dedupe_lock:
            self._inflight.pop(key, None)
            self._seen_ts[key] = time.time()

    # Note: This method is now async because it needs to acquire the lock.
    async def get_conversation_snapshot(self, history_idx: int) -> List[Tuple[str, str]]:
        async with self._history_lock:
            # Return a copy of the current history for the given index
            if history_idx in self.history:
                 return list(self.history[history_idx])
            return []

    async def append_to_history(self, history_idx: int, user_msg: str, assistant_msg: str) -> None:
        async with self._history_lock:
            if history_idx in self.history:
                self.history[history_idx].append(("user", user_msg))
                self.history[history_idx].append(("assistant", assistant_msg))

    def build_channel_messages(self, sender: str, answer: str) -> List[str]:
        fixed_prefix = f"@[{sender}] " if sender else ""
        return split_for_transport(
            text=answer,
            max_len=self.max_reply_chars,
            fixed_prefix=fixed_prefix,
            number_parts=True,
        )

    def build_dm_messages(self, answer: str) -> List[str]:
        return split_for_transport(
            text=answer,
            max_len=self.max_reply_chars,
            fixed_prefix="",
            number_parts=True,
        )

    # ---------------- Event handlers ----------------

    async def on_channel_msg(self, ev) -> None:
        if self.is_stale():
            if self.debug:
                print(f"[DBG] stale bot instance ignored channel event (gen={self.generation})")
            return

        p = ev.payload or {}
        if not isinstance(p, dict):
            return
        
        chan_idx = p.get("channel_idx")
        # Check if this message is on a channel we are monitoring
        if chan_idx not in self.monitored_channels:
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)
        # Use a default timestamp if missing to avoid dedupe issues
        sender_ts = p.get("sender_timestamp")
        if not isinstance(sender_ts, int):
            sender_ts = -1

        # 1. Deduplication (fast, locked)
        dedupe_key = await self.dedupe_enter("chan", chan_idx, sender_ts, body)
        if dedupe_key is None:
            return

        try:
            if self.debug:
                # Show channel name in debug print
                chan_name = self.monitored_channels.get(chan_idx, "unknown")
                print(f"[DBG] channel msg on '{chan_name}' (idx={chan_idx}) payload={p} gen={self.generation}")

            user_msg = self.extract_after_trigger(body)
            if not user_msg:
                return
            
            if user_msg.lower() == "ping":
                # Generate a reply containing network stats
                ping_reply = self.build_ping_reply(p, sender_name_from_text=sender)
                # Note: build_channel_messages already prefixes with @[sender]
                for out in self.build_channel_messages(sender, ping_reply):
                    await self.mesh.commands.send_chan_msg(chan_idx, out)
                return

            # 2. Prepare context and prompt (fast, unlocked)
            # Get history snapshot for THIS specific channel index
            hist_snapshot = await self.get_conversation_snapshot(chan_idx)
            conversation = hist_snapshot + [("user", user_msg)]

            requester_ctx = self.build_requester_context("chan", p, channel_idx=chan_idx, sender_name=sender)
            sys_prompt = self.effective_system_prompt(requester_ctx)

            # 3. Call LLM (SLOW, UNLOCKED)
            # This is the most important optimization: allowing concurrent LLM calls.
            try:
                answer = await self.llm.generate(sys_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            # 4. Update history (fast, locked)
            # Update history for THIS specific channel index
            await self.append_to_history(chan_idx, user_msg, answer)

            # 5. Send replies (relatively fast, unlocked)
            # Sending messages sequentially to maintain order of parts back to the source channel.
            for out in self.build_channel_messages(sender, answer):
                await self.mesh.commands.send_chan_msg(chan_idx, out)
        finally:
            # 6. Dedupe exit (fast, locked)
            await self.dedupe_exit(dedupe_key)

    async def on_dm_msg(self, ev) -> None:
        if self.is_stale():
            if self.debug:
                print(f"[DBG] stale bot instance ignored DM event (gen={self.generation})")
            return

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

        # 1. Deduplication (fast, locked)
        dedupe_key = await self.dedupe_enter("dm", -1, sender_ts, body)
        if dedupe_key is None:
            return

        try:
            if self.debug:
                print(f"[DBG] DM payload={p} bot_id={id(self)} gen={self.generation}")

            user_msg = self.extract_after_trigger(body)
            if not user_msg:
                return

            # Resolve destination before expensive LLM call
            dst = await self.resolve_dm_dst(p)
            if dst is None:
                # Last ditch effort to refresh contacts if not found.
                # This is a blocking call, but necessary if we can't reply.
                await self.refresh_contacts_best_effort()
                dst = await self.resolve_dm_dst(p)

            if dst is None:
                if self.debug:
                    print("[DBG] Could not resolve DM destination to full public_key; cannot reply.")
                return
            
            if user_msg.lower() == "ping":
                # Generate a reply containing network stats
                ping_reply = self.build_ping_reply(p, sender_name_from_text=sender)
                for out in self.build_dm_messages(ping_reply):
                    await self.mesh.commands.send_msg(dst, out)
                return

            # 2. Prepare context and prompt (fast, unlocked)
            # Use the dedicated DM history index
            hist_snapshot = await self.get_conversation_snapshot(DM_HISTORY_IDX)
            conversation = hist_snapshot + [("user", user_msg)]

            requester_ctx = self.build_requester_context("dm", p, sender_name=sender)
            sys_prompt = self.effective_system_prompt(requester_ctx)

            # 3. Call LLM (SLOW, UNLOCKED)
            try:
                answer = await self.llm.generate(sys_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            # 4. Update history (fast, locked)
            await self.append_to_history(DM_HISTORY_IDX, user_msg, answer)

            # 5. Send replies (relatively fast, unlocked)
            for out in self.build_dm_messages(answer):
                await self.mesh.commands.send_msg(dst, out)
        finally:
            # 6. Dedupe exit (fast, locked)
            await self.dedupe_exit(dedupe_key)


# ---------------------------
# connection
# ---------------------------

async def create_mesh_connection() -> MeshCore:
    transport = env_str("MESHCORE_TRANSPORT", "tcp").strip().lower()

    if transport == "tcp":
        host = env_str("MESHCORE_HOST", "")
        if not host:
            raise RuntimeError("Missing MESHCORE_HOST (required for MESHCORE_TRANSPORT=tcp)")
        port = env_int("MESHCORE_PORT", 5000)

        print(f"[INFO] MeshCore transport=tcp host={host} port={port}")
        mesh = await MeshCore.create_tcp(host, port, auto_reconnect=False)
        if mesh is None:
            raise RuntimeError("MeshCore.create_tcp() returned None")
        return mesh

    if transport == "serial":
        serial_port = env_str("MESHCORE_SERIAL_PORT", "")
        if not serial_port:
            raise RuntimeError("Missing MESHCORE_SERIAL_PORT (required for MESHCORE_TRANSPORT=serial)")
        baud = env_int("MESHCORE_SERIAL_BAUD", 115200)

        print(f"[INFO] MeshCore transport=serial port={serial_port} baud={baud}")

        # Try known factory methods for serial connection
        for factory_method in ["create_serial", "create_uart", "create_usb", "create_serial_port"]:
            if hasattr(MeshCore, factory_method):
                fn = getattr(MeshCore, factory_method)
                # Some methods might not take baud rate, but most do.
                try:
                    mesh = await fn(serial_port, baud, auto_reconnect=False)
                except TypeError:
                     # Fallback for methods that might not take baud
                    mesh = await fn(serial_port, auto_reconnect=False)
                
                if mesh is None:
                    raise RuntimeError(f"MeshCore.{factory_method}() returned None")
                return mesh

        raise RuntimeError(
            "Your meshcore package does not expose a known serial creation method. "
            "Run: python -c \"from meshcore import MeshCore; print([m for m in dir(MeshCore) if 'create' in m])\""
        )

    raise RuntimeError("MESHCORE_TRANSPORT must be one of: tcp | serial")


# ---------------------------
# run loop
# ---------------------------

_CURRENT_GENERATION = 0


def current_generation() -> int:
    return _CURRENT_GENERATION


async def run_bot_once(generation: int, llm: LLMClient) -> None:
    # Read comma-separated list of channels
    channel_names_str = env_str("MESHCORE_CHANNELS", "#avl-ai")
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    max_reply_chars = env_int("MAX_REPLY_CHARS", 140)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    debug = env_bool("DEBUG", False)
    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    include_requester_context = env_bool("INCLUDE_REQUESTER_CONTEXT", True)
    ping_reply_template = env_str("PING_REPLY_TEMPLATE", DEFAULT_PING_TEMPLATE)

    # LLM client creation is now moved to main()

    print("[INFO] connecting to MeshCore...")
    mesh = await create_mesh_connection()
    # Give the connection a moment to settle
    await asyncio.sleep(1.0)

    if mesh is None:
        raise RuntimeError("MeshCore connection failed: got None")

    await mesh.start_auto_message_fetching()

    # Find all requested channels
    want_names = {normalize_channel_name(n) for n in channel_names_str.split(",") if n.strip()}
    monitored_channels: Dict[int, str] = {}

    print(f"[INFO] Scanning for channels: {want_names} (max_scan={scan_max})")
    for i in range(scan_max):
        ev = await mesh.commands.get_channel(i)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if not isinstance(payload, dict):
            continue
        
        name_raw = payload.get("channel_name") or payload.get("name") or payload.get("chan_name") or ""
        norm_name = normalize_channel_name(str(name_raw))
        
        if norm_name and norm_name in want_names:
             # Store the original name for display purposes
             monitored_channels[i] = str(name_raw)
             print(f"[OK] Found required channel: '{name_raw}' at index {i}")

    if not monitored_channels:
         print(f"[ERROR] None of the requested channels ({want_names}) found in first {scan_max} slots.")
         await mesh.aclose()
         return

    bot = ChannelLLMBot(
        mesh=mesh,
        llm=llm,
        monitored_channels=monitored_channels,
        trigger=trigger,
        max_reply_chars=max_reply_chars,
        history_turns=history_turns,
        dedupe_window_s=dedupe_window_s,
        debug=debug,
        system_prompt=system_prompt,
        include_requester_context=include_requester_context,
        ping_reply_template=ping_reply_template,
        generation=generation,
        current_generation_ref=current_generation,
    )

    print(f"[BOOT] bot instance id={id(bot)} generation={generation}")

    mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
    mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

    await bot.refresh_contacts_best_effort()

    channel_list_str = ", ".join([f"{name}({idx})" for idx, name in monitored_channels.items()])
    print(f"[OK] Connected | listening on channels: {channel_list_str} | trigger='{trigger}'")
    print(f"[OK] Listening for DMs via CONTACT_MSG_RECV (trigger='{trigger}')")
    print(f"[CTX] INCLUDE_REQUESTER_CONTEXT={1 if include_requester_context else 0}")
    print(f"[CFG] PING_REPLY_TEMPLATE='{ping_reply_template}'")

    loop = asyncio.get_running_loop()
    disconnect_future: asyncio.Future[None] = loop.create_future()

    async def disconnected_handler(ev) -> None:
        print("[WARN] MeshCore disconnected event received")
        if not disconnect_future.done():
            disconnect_future.set_result(None)

    mesh.subscribe(EventType.DISCONNECTED, disconnected_handler)

    # Grab the first found channel index to use for health monitoring
    health_check_idx = next(iter(monitored_channels.keys()))

    async def health_monitor() -> None:
        while not disconnect_future.done():
            try:
                # Active health check on one of the monitored channels
                ev = await mesh.commands.get_channel(health_check_idx)
                if ev is None or ev.type == EventType.ERROR:
                    print(f"[WARN] health check failed on channel idx {health_check_idx}; forcing reconnect")
                    if not disconnect_future.done():
                        disconnect_future.set_result(None)
                    return
            except Exception as e:
                print(f"[WARN] health check exception: {e}")
                if not disconnect_future.done():
                    disconnect_future.set_result(None)
                return
            await asyncio.sleep(10)

    monitor_task = asyncio.create_task(health_monitor())

    try:
        await disconnect_future
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        print("[INFO] Closing MeshCore connection...")
        await mesh.aclose()
        # LLM client aclose() is now handled in main()


async def main() -> None:
    global _CURRENT_GENERATION

    # Initialize LLM client once, outside the reconnect loop.
    backend = env_str("LLM_BACKEND", "gemini").lower()
    print(f"[LLM] Initializing backend={backend}")

    llm: LLMClient
    if backend == "gemini":
        api_key = env_str("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
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
        raise RuntimeError("LLM_BACKEND must be one of: gemini | ollama | openai_compat")

    base_delay = env_int("RECONNECT_DELAY_S", 5)
    max_delay = env_int("RECONNECT_MAX_DELAY_S", 60)
    delay = base_delay

    try:
        while True:
            try:
                _CURRENT_GENERATION += 1
                gen = _CURRENT_GENERATION
                # Pass the persistent LLM client instance
                await run_bot_once(gen, llm)
                # If run_bot_once returns normally, reset delay
                delay = base_delay
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[WARN] bot session ended unexpectedly: {e}")

            print(f"[INFO] reconnecting in {delay}s...")
            await asyncio.sleep(delay)
            # Exponential backoff
            delay = min(delay * 2, max_delay)
    finally:
        # Ensure LLM client is closed on application exit
        print("[INFO] Closing LLM client...")
        await llm.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")
