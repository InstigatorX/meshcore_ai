#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ listens to channel messages AND direct messages

Features
- Connects over TCP or USB/Serial
- Owns reconnect logic itself (MeshCore internal auto_reconnect is disabled)
- Listens on MULTIPLE MeshCore channels defined in config.
- Also listens for direct messages (CONTACT_MSG_RECV)
- Triggers on "!ai ..." at start or after "NAME: " in channels.
- Channel replies use @[...] prefix.
- DMs do NOT require trigger and do NOT use @[...] prefix.
- De-dupes duplicate inbound packets.
- Resolves DM destinations via contacts cache (Fallback to prefix if needed).
- Splits long replies safely into numbered parts.
- Maintains conversation history per-user (based on pubkey_prefix).
- **VERBOSE DEBUGGING ENABLED.**
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
    return v in ("1", "true", "yes", "y", "on") if v else default

def normalize_channel_name(name: str) -> str:
    n = (name or "").strip()
    if n.startswith("#"): n = n[1:]
    return n.strip().lower()

def _split_with_budget(text: str, budget: int) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    if budget <= 0: raise ValueError("budget must be > 0")
    words = text.split(' ')
    chunks: List[str] = []
    current_chunk = ""
    for word in words:
        if not word: continue
        if len(word) > budget:
            if current_chunk: chunks.append(current_chunk); current_chunk = ""
            for i in range(0, len(word), budget): chunks.append(word[i:i+budget])
        elif len(current_chunk) + len(word) + (1 if current_chunk else 0) > budget:
            chunks.append(current_chunk); current_chunk = word
        else:
            current_chunk = (current_chunk + " " + word) if current_chunk else word
    if current_chunk: chunks.append(current_chunk)
    return chunks

def split_for_transport(text: str, max_len: int, fixed_prefix: str = "", number_parts: bool = True) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    fixed_len = len(fixed_prefix)
    if fixed_len + len(text) <= max_len: return [f"{fixed_prefix}{text}"]
    
    n_guess = max(2, (fixed_len + len(text) + max_len - 1) // max_len)
    while True:
        num_len = len(f"({n_guess}/{n_guess}) ") if n_guess > 1 else 0
        parts = _split_with_budget(text, max_len - fixed_len - num_len)
        if len(parts) <= n_guess: break
        n_guess = len(parts)

    out = []
    for i, part in enumerate(parts, 1):
        pfx = f"({i}/{len(parts)}) " if len(parts) > 1 else ""
        out.append(f"{fixed_prefix}{pfx}{part}"[:max_len])
    return out

# ---------------------------
# LLM Clients
# ---------------------------

class LLMClient:
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str: raise NotImplementedError
    async def aclose(self) -> None: pass

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        prompt = system_prompt + "\n\nConversation:\n" + "\n".join([f"{r}: {m}" for r, m in conversation]) + "\nassistant:"
        resp = await asyncio.to_thread(self.client.models.generate_content, model=self.model, contents=prompt)
        return str(getattr(resp, "text", "I couldn’t generate a response.")).strip()

class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, keep_alive: str = "5m"):
        self.base_url, self.model, self.keep_alive = base_url.rstrip("/"), model, keep_alive
        self._http = httpx.AsyncClient(timeout=60.0)
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        msgs = [{"role": "system", "content": system_prompt}] + [{"role": "assistant" if r == "assistant" else "user", "content": m} for r, m in conversation]
        r = await self._http.post(f"{self.base_url}/api/chat", json={"model": self.model, "messages": msgs, "stream": False, "keep_alive": self.keep_alive})
        return r.json().get("message", {}).get("content", "").strip()
    async def aclose(self) -> None: await self._http.aclose()

# ---------------------------
# Bot Logic
# ---------------------------

class ChannelLLMBot:
    def __init__(self, mesh: MeshCore, llm: LLMClient, monitored: Dict[int, str], trigger: str, max_reply_chars: int, history_turns: int, dedupe_window_s: float, debug: bool, system_prompt: str, include_ctx: bool, ping_tpl: str, generation: int, gen_ref):
        self.mesh, self.llm, self.monitored = mesh, llm, monitored
        self.trigger, self.max_reply_chars, self.history_turns = trigger, max_reply_chars, history_turns
        self.dedupe_window_s, self.debug, self.system_prompt = dedupe_window_s, debug, system_prompt
        self.include_ctx, self.ping_tpl, self.generation, self.gen_ref = include_ctx, ping_tpl, generation, gen_ref
        
        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)
        self.history: Dict[str, Deque[Tuple[str, str]]] = {}
        self._history_lock = asyncio.Lock()
        self._dedupe_lock = asyncio.Lock()
        self._seen_ts, self._inflight = {}, {}
        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey, self._contacts_by_prefix = {}, {}

    def is_stale(self) -> bool: return self.generation != self.gen_ref()

    async def upsert_contact(self, contact: Dict[str, Any]) -> None:
        pk = contact.get("public_key")
        if not isinstance(pk, str) or not pk.strip(): return
        pubkey = pk.strip().lower()
        prefix = pubkey[:12]
        async with self._contacts_lock:
            self._contacts_by_pubkey[pubkey] = contact
            self._contacts_by_prefix[prefix] = pubkey
        if self.debug:
            name = contact.get("name") or contact.get("alias") or ""
            print(f"[DBG] cached contact prefix={prefix} name={name}")

    async def on_contacts_event(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return
        candidates = []
        if isinstance(p.get("contacts"), list):
            for c in p["contacts"]:
                if isinstance(c, dict): candidates.append(c)
        if isinstance(p.get("public_key"), str): candidates.append(p)
        for c in candidates: await self.upsert_contact(c)

    async def resolve_dm_dst(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pk = payload.get("public_key")
        if isinstance(pk, str) and pk.strip(): return {"public_key": pk.strip()}
        prefix = payload.get("pubkey_prefix")
        if not isinstance(prefix, str) or not prefix.strip(): return None
        prefix = prefix.strip().lower()
        async with self._contacts_lock:
            pubkey = self._contacts_by_prefix.get(prefix)
            if not pubkey:
                for k in self._contacts_by_pubkey.keys():
                    if k.startswith(prefix): pubkey = k; break
        if pubkey: return {"public_key": pubkey}
        if self.debug: print(f"[DBG] Prefix {prefix} not in contacts; using prefix fallback.")
        return {"public_key": prefix}

    def build_ping_reply(self, payload: Dict[str, Any], sender: str = "") -> str:
        who = sender or payload.get("pubkey_prefix", "unknown")[:8]
        snr = payload.get("SNR", 0)
        q = "Poor"
        if snr > 10: q = "Ok"
        if snr > 20: q = "Good"
        if snr > 30: q = "Great"
        hops = max(0, payload.get("path_len", 1) - 1)
        stats = f"SNR: {snr:.0f}dB ({q}) | Hops: {hops}"
        try: return self.ping_tpl.format(who=who, stats=stats).replace("[]", "").strip()
        except: return f"🤖 Ack {who}\n[{stats}]"

    def build_ctx(self, scope: str, payload: Dict[str, Any], chan_idx: int = -1, sender: str = "") -> str:
        if not self.include_ctx: return ""
        lines = [f"Requester context:\n- scope: {scope}"]
        if scope == "chan": lines.append(f"- channel: {self.monitored.get(chan_idx, 'unknown')}")
        if sender: lines.append(f"- sender_name: {sender}")
        for f in ["pubkey_prefix", "SNR", "path_len", "sender_timestamp"]:
            v = payload.get(f)
            if v is not None: lines.append(f"- {f}: {v}")
        return "\n".join(lines)

    async def get_hist(self, user_id: str) -> List[Tuple[str, str]]:
        async with self._history_lock:
            if user_id not in self.history: self.history[user_id] = deque(maxlen=self.history_turns * 2)
            return list(self.history[user_id])

    async def add_hist(self, user_id: str, u: str, a: str) -> None:
        async with self._history_lock:
            if user_id in self.history: self.history[user_id].append(("user", u)), self.history[user_id].append(("assistant", a))

    def split_name(self, text: str) -> Tuple[str, str]:
        t = (text or "").strip()
        if ": " in t:
            n, b = t.split(": ", 1)
            if 0 < len(n) <= 40: return n.strip(), b.strip()
        return "", t

    async def dedupe(self, scope: str, ch_idx: int, ts: int, body: str) -> Optional[tuple]:
        key = (scope, ch_idx, ts, body)
        now = time.time()
        async with self._dedupe_lock:
            self._seen_ts = {k: v for k, v in self._seen_ts.items() if now - v < self.dedupe_window_s}
            self._inflight = {k: v for k, v in self._inflight.items() if now - v < 30.0}
            if key in self._seen_ts or key in self._inflight: return None
            self._inflight[key] = now
            return key

    async def _safe_send(self, is_dm: bool, dst_idx: Any, text: str):
        if is_dm:
            for m in ["send_msg", "send_direct_msg", "send_priv_msg"]:
                if hasattr(self.mesh.commands, m):
                    try: 
                        await getattr(self.mesh.commands, m)(dst_idx, text)
                        if self.debug: print(f"[DBG] DM sent via {m}")
                        return
                    except Exception as e: 
                        if self.debug: print(f"[DBG] DM {m} attempt failed: {e}")
            print("[ERROR] No working DM method found in meshcore library.")
        else:
            await self.mesh.commands.send_chan_msg(dst_idx, text)

    async def on_channel_msg(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return
        chan_idx = p.get("channel_idx")
        if chan_idx not in self.monitored: return
        
        sender, body = self.split_name(p.get("text", ""))
        dedupe_key = await self.dedupe("chan", chan_idx, p.get("sender_timestamp", -1), body)
        if not dedupe_key: return

        try:
            if self.debug: print(f"[DBG] channel msg payload={p} gen={self.generation}")
            
            if body.strip().lower() == "ping":
                for out in split_for_transport(self.build_ping_reply(p, sender), self.max_reply_chars, f"@[{sender}] "):
                    await self._safe_send(False, chan_idx, out)
                return

            user_msg = (self.trigger_re.split(body)[-1]).strip() if self.trigger_re.search(body) else None
            if not user_msg: return
            
            user_id = p.get("pubkey_prefix") or sender or "unknown_chan"
            hist = await self.get_hist(user_id)
            sys = f"{self.system_prompt}\n\n{self.build_ctx('chan', p, chan_idx, sender)}"
            
            if self.debug: print(f"[DBG] Channel calling LLM for user {user_id}")
            try: answer = await self.llm.generate(sys, hist + [("user", user_msg)])
            except Exception as e: answer = f"LLM error: {e}"
            
            await self.add_hist(user_id, user_msg, answer)
            for out in split_for_transport(answer, self.max_reply_chars, f"@[{sender}] "):
                await self._safe_send(False, chan_idx, out)
        finally: await self.dedupe_exit(dedupe_key)

    async def on_dm_msg(self, ev) -> None:
        if self.is_stale(): return
        p = ev.payload or {}
        if not isinstance(p, dict): return
        
        body = (p.get("text") or p.get("body") or "").strip()
        dedupe_key = await self.dedupe("dm", -1, p.get("sender_timestamp", -1), body)
        if not dedupe_key: return

        try:
            if self.debug: print(f"[DBG] DM payload={p} bot_id={id(self)} gen={self.generation}")
            
            dst = await self.resolve_dm_dst(p)
            if not dst: return

            if body.lower() == "ping":
                for out in split_for_transport(self.build_ping_reply(p), self.max_reply_chars):
                    await self._safe_send(True, dst, out)
                return

            user_id = p.get("pubkey_prefix") or p.get("public_key") or "unknown_dm"
            hist = await self.get_hist(user_id)
            sys = f"{self.system_prompt}\n\n{self.build_ctx('dm', p)}"
            
            if self.debug: print(f"[DBG] DM calling LLM for user {user_id}")
            try: answer = await self.llm.generate(sys, hist + [("user", body)])
            except Exception as e: answer = f"LLM error: {e}"
            
            await self.add_hist(user_id, body, answer)
            for out in split_for_transport(answer, self.max_reply_chars):
                await self._safe_send(True, dst, out)
        finally: await self.dedupe_exit(dedupe_key)

    async def dedupe_exit(self, key):
        async with self._dedupe_lock:
            self._inflight.pop(key, None)
            self._seen_ts[key] = time.time()

# ---------------------------
# Connection & Loop
# ---------------------------

async def create_mesh():
    t = env_str("MESHCORE_TRANSPORT", "tcp").lower()
    if t == "tcp": return await MeshCore.create_tcp(env_str("MESHCORE_HOST", ""), env_int("MESHCORE_PORT", 5000), auto_reconnect=False)
    p, b = env_str("MESHCORE_SERIAL_PORT", ""), env_int("MESHCORE_SERIAL_BAUD", 115200)
    for m in ["create_serial", "create_uart", "create_usb"]:
        if hasattr(MeshCore, m): return await getattr(MeshCore, m)(p, b, auto_reconnect=False)
    raise RuntimeError("No serial transport found")

async def run_bot(gen, llm):
    mesh = await create_mesh()
    if not mesh: raise RuntimeError("Connection failed")
    try:
        await asyncio.sleep(1); await mesh.start_auto_message_fetching()
        wants = {normalize_channel_name(n) for n in env_str("MESHCORE_CHANNELS", "#avl-ai").split(",") if n.strip()}
        monitored = {}
        for i in range(env_int("CHANNEL_SCAN_MAX", 16)):
            ev = await mesh.commands.get_channel(i)
            if ev.type != EventType.ERROR:
                n = ev.payload.get("channel_name") or ev.payload.get("name") or ""
                if normalize_channel_name(n) in wants: monitored[i] = n
        if not monitored: raise RuntimeError(f"No channels found for {wants}")

        bot = ChannelLLMBot(mesh, llm, monitored, env_str("AI_TRIGGER", "!ai"), env_int("MAX_REPLY_CHARS", 140), env_int("HISTORY_TURNS", 6), env_float("DEDUPE_WINDOW_S", 3.0), env_bool("DEBUG", False), env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT), env_bool("INCLUDE_REQUESTER_CONTEXT", True), env_str("PING_REPLY_TEMPLATE", DEFAULT_PING_TEMPLATE), gen, lambda: _CURRENT_GENERATION)
        
        mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
        mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
        mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)
        if hasattr(EventType, "DIRECT_MSG_RECV"): mesh.subscribe(EventType.DIRECT_MSG_RECV, bot.on_dm_msg)
        
        await bot.refresh_contacts_best_effort()
        print(f"[OK] Bot Gen {gen} monitoring: {list(monitored.values())}")
        
        exit_event = asyncio.get_running_loop().create_future()
        mesh.subscribe(EventType.DISCONNECTED, lambda _: not exit_event.done() and exit_event.set_result(None))

        async def health():
            while not exit_event.done():
                try: 
                    await asyncio.wait_for(mesh.commands.get_channel(next(iter(monitored))), 5.0)
                    await asyncio.sleep(15)
                except: exit_event.set_result(None); break
        
        hc = asyncio.create_task(health()); await exit_event; hc.cancel()
    finally: await mesh.aclose()

_CURRENT_GENERATION = 0
async def main():
    global _CURRENT_GENERATION
    b = env_str("LLM_BACKEND", "gemini").lower()
    if b == "gemini": llm = GeminiClient(env_str("GEMINI_API_KEY", ""), env_str("GEMINI_MODEL", "gemini-3-flash-preview"))
    elif b == "ollama": llm = OllamaClient(env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), env_str("OLLAMA_MODEL", "llama3.2:latest"))
    else: llm = OpenAICompatClient(env_str("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"), env_str("LOCAL_LLM_MODEL", "local-model"))

    delay = env_int("RECONNECT_DELAY_S", 5)
    while True:
        try: _CURRENT_GENERATION += 1; await run_bot(_CURRENT_GENERATION, llm); delay = env_int("RECONNECT_DELAY_S", 5)
        except Exception as e:
            print(f"[WARN] Session failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay); delay = min(delay * 2, env_int("RECONNECT_MAX_DELAY_S", 60))

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
