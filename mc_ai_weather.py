#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ listens to multiple channels AND direct messages
+ Scheduled daily weather broadcasts and on-demand `!weather [location]` fetching.
+ Includes today's forecast (High/Low/Rain chance)
+ Configurable polling for US National Weather Service (NWS) severe weather alerts.

DM replies require a destination with full 'public_key'. We resolve DM sender via
pubkey_prefix by caching contacts.

Also fixes a race where duplicate inbound packets could invoke the LLM twice by
locking the dedupe check.
"""

import asyncio
import os
import re
import time
import urllib.parse
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Set

import httpx
from meshcore import MeshCore, EventType

# Gemini optional
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
    for tok in re.split(r"(\s+)", text):  # keep whitespace
        if len(cur) + len(tok) <= max_len:
            cur += tok
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = tok
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


async def resolve_channels(mesh: MeshCore, channel_names: List[str], max_channels: int = 16) -> Dict[int, str]:
    """Finds the index of multiple channel names. Returns a dict of {idx: normalized_name}."""
    want = {normalize_channel_name(c) for c in channel_names if c.strip()}
    found: Dict[int, str] = {}
    
    for idx in range(max_channels):
        ev = await mesh.commands.get_channel(idx)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if not isinstance(payload, dict):
            continue
            
        got_raw = payload.get("channel_name") or payload.get("name") or payload.get("chan_name") or ""
        got = normalize_channel_name(str(got_raw))
        
        if got in want:
            found[idx] = got

    missing = want - set(found.values())
    if missing:
        print(f"[WARN] Could not find requested channels: {missing} in first {max_channels} slots")
        
    if not found:
        raise RuntimeError(f"None of the requested channels {channel_names} were found on this node!")
        
    return found


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant replying over a low-bandwidth MeshCore channel. "
    "Keep replies short and directly useful (prefer 1–3 sentences). "
    "If uncertain, say so briefly."
)


# ---------------------------
# LLM Clients
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
        channel_map: Dict[int, str],
        trigger: str,
        max_reply_chars: int,
        history_turns: int,
        dedupe_window_s: float,
        debug: bool,
        system_prompt: str,
        weather_location: str,
        weather_times: List[str],
        weather_trigger: str,
        weather_units: str,
    ):
        self.mesh = mesh
        self.llm = llm
        self.channel_map = channel_map  # {idx: normalized_name}
        self.trigger = trigger
        self.max_reply_chars = max_reply_chars
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s
        self.system_prompt = system_prompt

        self.weather_location = weather_location
        self.weather_times = weather_times
        self.weather_trigger = weather_trigger.strip()
        self.weather_units = weather_units.upper()
        self._seen_alerts: Set[str] = set()

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(trigger)}(\s+|$)", re.IGNORECASE)

        # Separate history per channel index
        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            idx: deque(maxlen=history_turns * 2) for idx in channel_map.keys()
        }

        # serialize LLM calls and mesh send bursts
        self._llm_lock = asyncio.Lock()

        # dedupe is separate and must be locked to avoid race duplicates
        self._dedupe_lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, int, str], float] = {}  # (scope, ch, sender_ts, body) -> time

        # Contacts cache for DM replies
        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}  # prefix -> pubkey

    # ---------------- Contacts ----------------

    async def upsert_contact(self, contact: Dict[str, Any]) -> None:
        pk = contact.get("public_key")
        if not isinstance(pk, str) or not pk.strip():
            return
        pubkey = pk.strip().lower()
        prefix = pubkey[:12]  # matches your pubkey_prefix length (e.g., '2adad8233a4d')

        async with self._contacts_lock:
            self._contacts_by_pubkey[pubkey] = contact
            self._contacts_by_prefix.setdefault(prefix, pubkey)

        if self.debug:
            name = contact.get("name") or contact.get("alias") or ""
            print(f"[DBG] cached contact pubkey_prefix={prefix} name={name}")

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

    # ---------------- Weather & Alerts Logic ----------------

    async def fetch_weather(self, location: str) -> str:
        """Fetches current weather and today's forecast from wttr.in using JSON format."""
        if not location:
            return "No weather location specified."
        
        loc_encoded = urllib.parse.quote(location)
        url = f"https://wttr.in/{loc_encoded}?format=j1"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            current = data.get("current_condition", [{}])[0]
            today = data.get("weather", [{}])[0]

            # Current Conditions
            cond = current.get("weatherDesc", [{"value": "Unknown"}])[0].get("value", "Unknown")
            temp_f = current.get("temp_F", "?")
            temp_c = current.get("temp_C", "?")
            wind_mph = current.get("windspeedMiles", "?")
            wind_kmh = current.get("windspeedKmph", "?")
            hum = current.get("humidity", "?")

            # Today's Forecast
            high_f = today.get("maxtempF", "?")
            low_f = today.get("mintempF", "?")
            high_c = today.get("maxtempC", "?")
            low_c = today.get("mintempC", "?")

            # Precipitation Chances (find the highest chance across all hours today)
            rain_chances = [int(h.get("chanceofrain", "0")) for h in today.get("hourly", [])]
            snow_chances = [int(h.get("chanceofsnow", "0")) for h in today.get("hourly", [])]
            max_rain = max(rain_chances) if rain_chances else 0
            max_snow = max(snow_chances) if snow_chances else 0

            # Title case the location to make it look nicer (e.g. asheville -> Asheville)
            display_loc = location.title()

            if self.weather_units == "C":
                msg = f"🌤️ {display_loc}: {cond} {temp_c}°C (Wind {wind_kmh}km/h, Hum {hum}%). " \
                      f"Today: High {high_c}°C / Low {low_c}°C"
            else:
                msg = f"🌤️ {display_loc}: {cond} {temp_f}°F (Wind {wind_mph}mph, Hum {hum}%). " \
                      f"Today: High {high_f}°F / Low {low_f}°F"

            # Append significant precipitation chance
            if max_snow > 0:
                msg += f", Snow {max_snow}%"
            elif max_rain > 0:
                msg += f", Rain {max_rain}%"

            return msg

        except Exception as e:
            if self.debug:
                print(f"[DBG] Weather fetch error for {location}: {e}")
            return f"Could not find weather for '{location}' (API Error)."

    async def scheduled_weather_loop(self) -> None:
        """Broadcasts weather at specifically configured times of day."""
        if not self.weather_times or not self.weather_location:
            return

        parsed_times = []
        for t_str in self.weather_times:
            parts = t_str.split(":")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                parsed_times.append((int(parts[0]), int(parts[1])))

        if not parsed_times:
            return

        last_sent_min = -1

        while True:
            now = time.localtime()
            current_time = (now.tm_hour, now.tm_min)

            if current_time in parsed_times and now.tm_min != last_sent_min:
                last_sent_min = now.tm_min
                
                if self.debug:
                    print(f"[DBG] Triggering scheduled weather for {current_time}")
                
                ans = await self.fetch_weather(self.weather_location)
                
                # Send to ALL monitored channels
                async with self._llm_lock:
                    parts = chunk_text(ans, self.max_reply_chars)
                    for ch_idx in self.channel_map.keys():
                        for i, part in enumerate(parts, start=1):
                            msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                            await self.mesh.commands.send_chan_msg(ch_idx, msg)
                            await asyncio.sleep(2) # Prevent overwhelming network with burst

            await asyncio.sleep(10)

    async def weather_alerts_loop(self, zones: str, interval_m: float) -> None:
        """Polls the NWS API periodically for new weather alerts."""
        if not zones or interval_m <= 0:
            return

        interval_s = interval_m * 60.0
        headers = {"User-Agent": "MeshCore-Weather-Alert-Bot/1.0 (https://github.com/)"}
        url = f"https://api.weather.gov/alerts/active?zone={urllib.parse.quote(zones)}"

        while True:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                
                features = data.get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    alert_id = props.get("id")
                    
                    if not alert_id or alert_id in self._seen_alerts:
                        continue
                    
                    self._seen_alerts.add(alert_id)
                    
                    if len(self._seen_alerts) > 500:
                        self._seen_alerts.clear()
                        self._seen_alerts.add(alert_id)

                    headline = props.get("headline") or props.get("event") or "Unknown Severe Weather Alert"
                    severity = props.get("severity", "Unknown")
                    msg = f"🚨 ALERT ({severity}): {headline}"

                    if self.debug:
                        print(f"[DBG] New Alert Found: {msg}")

                    # Broadcast alert to all mapped channels
                    async with self._llm_lock:
                        parts = chunk_text(msg, self.max_reply_chars)
                        for ch_idx in self.channel_map.keys():
                            for i, part in enumerate(parts, start=1):
                                out_msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                                await self.mesh.commands.send_chan_msg(ch_idx, out_msg)
                                await asyncio.sleep(2)

            except Exception as e:
                if self.debug:
                    print(f"[DBG] Weather alert polling error: {e}")

            await asyncio.sleep(interval_s)

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
        return b[idx + len(self.trigger) :].strip(" \t:,-")

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

    def build_conversation(self, ch_idx: int, user_text: str) -> List[Tuple[str, str]]:
        hist = list(self.history[ch_idx])
        return hist + [("user", user_text)]

    def format_chan_reply(self, sender: str, msg: str) -> str:
        msg = (msg or "").strip()
        if not msg:
            return ""
        return f"@[{sender}] {msg}" if sender else msg

    def format_dm_reply(self, msg: str) -> str:
        return (msg or "").strip()

    # ---------------- Event handlers ----------------

    async def on_channel_msg(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return
            
        ch_idx = p.get("channel_idx")
        if ch_idx not in self.channel_map:
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)
        sender_ts = p.get("sender_timestamp")
        if not isinstance(sender_ts, int):
            sender_ts = -1

        if await self.dedupe_drop("chan", ch_idx, sender_ts, body):
            return

        # On-Demand Weather Check
        body_lower = body.lower()
        w_trigger = self.weather_trigger.lower()
        if body_lower == w_trigger or body_lower.startswith(w_trigger + " "):
            req_loc = body[len(self.weather_trigger):].strip()
            target_loc = req_loc if req_loc else self.weather_location
            
            ans = await self.fetch_weather(target_loc)
            
            async with self._llm_lock:
                parts = chunk_text(ans, self.max_reply_chars)
                for i, part in enumerate(parts, start=1):
                    msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                    out = self.format_chan_reply(sender, msg)
                    if out:
                        await self.mesh.commands.send_chan_msg(ch_idx, out)
            return

        # LLM Trigger Check
        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            if user.lower() == "ping":
                out = self.format_chan_reply(sender, "pong")
                if out:
                    await self.mesh.commands.send_chan_msg(ch_idx, out)
                return

            self.history[ch_idx].append(("user", user))
            conversation = self.build_conversation(ch_idx, user)

            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[ch_idx].append(("assistant", answer))

            parts = chunk_text(answer, self.max_reply_chars)
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_chan_reply(sender, msg)
                if out:
                    await self.mesh.commands.send_chan_msg(ch_idx, out)

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

        dst = self.resolve_dm_dst(p)
        if dst is None:
            await self.refresh_contacts_best_effort()
            dst = self.resolve_dm_dst(p)

        if dst is None:
            if self.debug:
                print("[DBG] Could not resolve DM destination to full public_key; cannot reply.")
            return

        # On-Demand Weather Check
        body_lower = body.lower()
        w_trigger = self.weather_trigger.lower()
        if body_lower == w_trigger or body_lower.startswith(w_trigger + " "):
            req_loc = body[len(self.weather_trigger):].strip()
            target_loc = req_loc if req_loc else self.weather_location
            
            ans = await self.fetch_weather(target_loc)
            
            async with self._llm_lock:
                parts = chunk_text(ans, self.max_reply_chars)
                for i, part in enumerate(parts, start=1):
                    msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                    out = self.format_dm_reply(msg)
                    if out:
                        await self.mesh.commands.send_msg(dst, out)
            return

        # LLM Trigger Check
        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            if user.lower() == "ping":
                out = self.format_dm_reply("pong")
                if out:
                    await self.mesh.commands.send_msg(dst, out)
                return

            dummy_idx = 999 
            if dummy_idx not in self.history:
                self.history[dummy_idx] = deque(maxlen=self.history[list(self.channel_map.keys())[0]].maxlen)
            
            self.history[dummy_idx].append(("user", user))
            conversation = self.build_conversation(dummy_idx, user)
            
            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[dummy_idx].append(("assistant", answer))

            parts = chunk_text(answer, self.max_reply_chars)
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_dm_reply(msg)
                if out:
                    await self.mesh.commands.send_msg(dst, out)


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


async def main() -> None:
    channels_raw = env_str("MESHCORE_CHANNELS", "#avl-ai")
    target_channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")
    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    # Weather configurations
    weather_location = env_str("WEATHER_LOCATION", "")
    weather_times_raw = env_str("WEATHER_SCHEDULED_TIMES", "")
    weather_times = [t.strip() for t in weather_times_raw.split(",") if t.strip()]
    weather_trigger = env_str("WEATHER_TRIGGER", "!weather").strip()
    weather_units = env_str("WEATHER_UNITS", "F")  # "F" or "C"
    
    # Alert configurations
    alerts_zones = env_str("WEATHER_ALERTS_NWS_ZONES", "")
    alerts_interval_m = env_float("WEATHER_ALERTS_POLL_INTERVAL_M", 15.0)

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

    # Resolve all requested channels into a dictionary {index: name}
    channel_map = await resolve_channels(mesh, target_channels, max_channels=scan_max)

    print("[OK] Channel map:")
    for idx, name in channel_map.items():
        print(f"  idx={idx} -> {name}")

    bot = ChannelLLMBot(
        mesh=mesh,
        llm=llm,
        channel_map=channel_map,
        trigger=trigger,
        max_reply_chars=max_reply_chars,
        history_turns=history_turns,
        dedupe_window_s=dedupe_window_s,
        debug=debug,
        system_prompt=system_prompt,
        weather_location=weather_location,
        weather_times=weather_times,
        weather_trigger=weather_trigger,
        weather_units=weather_units
    )

    # Contacts/cache feeders
    mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
    mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)

    # Message listeners
    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

    # Try to warm contacts cache
    await bot.refresh_contacts_best_effort()

    # Start weather schedule background task if configured
    if weather_times and weather_location:
        asyncio.create_task(bot.scheduled_weather_loop())
        print(f"[OK] Scheduled weather enabled: Location='{weather_location}', Times={weather_times}, Units={weather_units}")
        
    # Start NWS weather alerts polling task if configured
    if alerts_zones and alerts_interval_m > 0:
        asyncio.create_task(bot.weather_alerts_loop(alerts_zones, alerts_interval_m))
        print(f"[OK] NWS weather alerts polling enabled: Zones='{alerts_zones}', Interval={alerts_interval_m}m")

    print(f"[OK] Connected | listening on channels {list(channel_map.values())} | AI trigger='{trigger}'")
    print(f"[OK] Listening for DMs via CONTACT_MSG_RECV")
    print(f"[OK] On-demand weather enabled. Trigger='{weather_trigger}' (e.g. '{weather_trigger}' or '{weather_trigger} Paris')")
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

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    asyncio.run(main())
