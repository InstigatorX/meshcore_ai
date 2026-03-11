#!/usr/bin/env python3
"""
MeshCore weather + AI bot

Features
- TCP or serial MeshCore transport
- App-managed reconnect loop (MeshCore auto_reconnect disabled)
- Separate AI channels and Weather channels
- DM support for all commands
- Global commands: !help, !ping
- AI command: !ai ...
- Weather command: !weather [location]
- Scheduled weather broadcasts
- NWS severe weather alert polling
- Per-channel AI history
- Contact cache for DM replies
- Idle-based reconnect health monitor (uses last received packet time)

Environment
  MESHCORE_TRANSPORT=tcp|serial             default: tcp
  MESHCORE_HOST=...                         required for tcp
  MESHCORE_PORT=5000
  MESHCORE_SERIAL_PORT=/dev/ttyACM0         required for serial
  MESHCORE_SERIAL_BAUD=115200
  MESHCORE_AI_CHANNELS=#avl-ai
  MESHCORE_WEATHER_CHANNELS=#weather-avl
  CHANNEL_SCAN_MAX=16

  AI_TRIGGER=!ai
  PING_TRIGGER=!ping
  HELP_TRIGGER=!help
  WEATHER_TRIGGER=!weather
  PING_TEMPLATE="pong [SNR: {snr}, RSSI: {rssi}dBm, Hops: {hops}]"

  MAX_REPLY_CHARS=180
  HISTORY_TURNS=6
  DEDUPE_WINDOW_S=3.0
  DEBUG=0

  WEATHER_LOCATION=Asheville, NC
  WEATHER_SCHEDULED_TIMES=07:30,18:00
  WEATHER_UNITS=F
  WEATHER_ALERTS_NWS_ZONES=NCZ053,NCC021
  WEATHER_ALERTS_POLL_INTERVAL_M=15

  HEALTH_IDLE_TIMEOUT_S=300
  HEALTHCHECK_INTERVAL_S=10
  RECONNECT_DELAY_S=5
  RECONNECT_MAX_DELAY_S=60

  LLM_BACKEND=gemini|ollama|openai_compat
  SYSTEM_PROMPT=...

  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-3-flash-preview

  OLLAMA_BASE_URL=http://127.0.0.1:11434
  OLLAMA_MODEL=llama3.2:latest
  OLLAMA_KEEP_ALIVE=5m

  LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
  LOCAL_LLM_MODEL=local-model
  LOCAL_LLM_API_KEY=
  LOCAL_LLM_TEMPERATURE=0.3
"""

import asyncio
import os
import re
import time
import urllib.parse
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
from meshcore import EventType, MeshCore

try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # noqa: N816


# ---------------------------
# helpers
# ---------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant replying over a low-bandwidth MeshCore channel. "
    "Keep replies short and directly useful (prefer 1–3 sentences). "
    "If uncertain, say so briefly."
)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return int(v) if v else default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def normalize_channel_name(name: str) -> str:
    n = (name or "").strip()
    return n[1:].strip().lower() if n.startswith("#") else n.lower()


def chunk_text(text: str, max_len: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    out: List[str] = []
    cur = ""
    for tok in re.split(r"(\s+)", text):
        if len(cur) + len(tok) <= max_len:
            cur += tok
        else:
            if cur.strip():
                out.append(cur.strip())
            cur = tok
    if cur.strip():
        out.append(cur.strip())
    return out


async def resolve_channels(mesh: MeshCore, names: List[str], max_channels: int) -> Dict[int, str]:
    wanted = {normalize_channel_name(x) for x in names if x.strip()}
    found: Dict[int, str] = {}

    for idx in range(max_channels):
        ev = await mesh.commands.get_channel(idx)
        if ev.type == EventType.ERROR:
            continue
        payload = ev.payload or {}
        if not isinstance(payload, dict):
            continue
        raw = payload.get("channel_name") or payload.get("name") or payload.get("chan_name") or ""
        got = normalize_channel_name(str(raw))
        if got in wanted:
            found[idx] = got

    if not found:
        raise RuntimeError(f"None of the requested channels were found: {names}")

    missing = wanted - set(found.values())
    if missing:
        print(f"[WARN] Missing configured channels: {sorted(missing)}")

    return found


async def create_mesh_connection() -> MeshCore:
    transport = env_str("MESHCORE_TRANSPORT", "tcp").lower()

    if transport == "tcp":
        host = env_str("MESHCORE_HOST", "")
        if not host:
            raise RuntimeError("Missing MESHCORE_HOST")
        port = env_int("MESHCORE_PORT", 5000)
        print(f"[INFO] MeshCore transport=tcp host={host} port={port}")
        mesh = await MeshCore.create_tcp(host, port, auto_reconnect=False)
        if mesh is None:
            raise RuntimeError("MeshCore.create_tcp() returned None")
        return mesh

    if transport == "serial":
        port = env_str("MESHCORE_SERIAL_PORT", "")
        if not port:
            raise RuntimeError("Missing MESHCORE_SERIAL_PORT")
        baud = env_int("MESHCORE_SERIAL_BAUD", 115200)
        print(f"[INFO] MeshCore transport=serial port={port} baud={baud}")

        for factory in ("create_serial", "create_uart", "create_usb", "create_serial_port"):
            if hasattr(MeshCore, factory):
                fn = getattr(MeshCore, factory)
                try:
                    mesh = await fn(port, baud, auto_reconnect=False)
                except TypeError:
                    mesh = await fn(port, auto_reconnect=False)
                if mesh is None:
                    raise RuntimeError(f"MeshCore.{factory}() returned None")
                return mesh

        raise RuntimeError("No supported MeshCore serial factory method found")

    raise RuntimeError("MESHCORE_TRANSPORT must be tcp or serial")


async def close_mesh(mesh: MeshCore) -> None:
    if hasattr(mesh, "aclose"):
        await mesh.aclose()
    elif hasattr(mesh, "close"):
        mesh.close()


# ---------------------------
# llm backends
# ---------------------------

class LLMClient:
    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        if genai is None:
            raise RuntimeError("google-genai not installed")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        lines = [system_prompt, "", "Conversation:"]
        for role, msg in conversation:
            lines.append(f"{role}: {msg}")
        lines.append("assistant:")
        prompt = "\n".join(lines)

        def _call() -> str:
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            txt = getattr(resp, "text", None)
            return str(txt).strip() if txt else ""

        out = await asyncio.to_thread(_call)
        return out or "I couldn’t generate a response."


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, keep_alive: str = "5m", timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": msg})

        r = await self.http.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive},
        )
        r.raise_for_status()
        data = r.json()
        return ((data.get("message") or {}).get("content") or "").strip() or "I couldn’t generate a response."

    async def aclose(self) -> None:
        await self.http.aclose()


class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_key: Optional[str], temperature: float, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def generate(self, system_prompt: str, conversation: List[Tuple[str, str]]) -> str:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [{"role": "system", "content": system_prompt}]
        for role, msg in conversation:
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": msg})

        r = await self.http.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages, "temperature": self.temperature},
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return "I couldn’t generate a response."
        return str(((choices[0].get("message") or {}).get("content")) or "").strip() or "I couldn’t generate a response."

    async def aclose(self) -> None:
        await self.http.aclose()


def build_llm() -> LLMClient:
    backend = env_str("LLM_BACKEND", "gemini").lower()
    if backend == "gemini":
        api_key = env_str("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        return GeminiClient(api_key, env_str("GEMINI_MODEL", "gemini-3-flash-preview"))
    if backend == "ollama":
        return OllamaClient(
            env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            env_str("OLLAMA_MODEL", "llama3.2:latest"),
            env_str("OLLAMA_KEEP_ALIVE", "5m"),
        )
    if backend == "openai_compat":
        return OpenAICompatClient(
            env_str("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
            env_str("LOCAL_LLM_MODEL", "local-model"),
            env_str("LOCAL_LLM_API_KEY", "") or None,
            env_float("LOCAL_LLM_TEMPERATURE", 0.3),
        )
    raise RuntimeError("LLM_BACKEND must be one of: gemini | ollama | openai_compat")


# ---------------------------
# bot
# ---------------------------

class MeshBot:
    def __init__(
        self,
        mesh: MeshCore,
        llm: LLMClient,
        ai_channels: Dict[int, str],
        weather_channels: Dict[int, str],
    ):
        self.mesh = mesh
        self.llm = llm
        self.ai_channels = ai_channels
        self.weather_channels = weather_channels

        self.trigger = env_str("AI_TRIGGER", "!ai").strip()
        self.ping_trigger = env_str("PING_TRIGGER", "!ping").strip()
        self.help_trigger = env_str("HELP_TRIGGER", "!help").strip()
        self.weather_trigger = env_str("WEATHER_TRIGGER", "!weather").strip()
        self.ping_template = env_str("PING_TEMPLATE", "pong [SNR: {snr}, RSSI: {rssi}dBm, Hops: {hops}]")

        self.max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
        self.debug = env_bool("DEBUG", False)
        self.system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.weather_location = env_str("WEATHER_LOCATION", "")
        self.weather_units = env_str("WEATHER_UNITS", "F").upper()

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(self.trigger)}(\s+|$)", re.IGNORECASE)

        history_turns = env_int("HISTORY_TURNS", 6)
        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            idx: deque(maxlen=history_turns * 2) for idx in ai_channels.keys()
        }
        self.dm_history: Deque[Tuple[str, str]] = deque(maxlen=history_turns * 2)

        self._llm_lock = asyncio.Lock()

        self.dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
        self._dedupe_lock = asyncio.Lock()
        self._seen: Dict[Tuple[str, int, int, str], float] = {}

        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}

        self._seen_alerts: Set[str] = set()
        self.last_rx_time = time.time()

    # ----- contacts -----

    async def upsert_contact(self, contact: Dict[str, Any]) -> None:
        pk = contact.get("public_key")
        if not isinstance(pk, str) or not pk.strip():
            return
        pubkey = pk.strip().lower()
        prefix = pubkey[:12]

        async with self._contacts_lock:
            self._contacts_by_pubkey[pubkey] = contact
            self._contacts_by_prefix[prefix] = pubkey

        if self.debug:
            name = contact.get("name") or contact.get("alias") or ""
            print(f"[DBG] cached contact prefix={prefix} name={name}")

    async def on_contacts_event(self, ev) -> None:
        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        candidates: List[Dict[str, Any]] = []
        if isinstance(p.get("contacts"), list):
            candidates.extend([c for c in p["contacts"] if isinstance(c, dict)])
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
            for pk2 in self._contacts_by_pubkey:
                if pk2.startswith(prefix):
                    pubkey = pk2
                    break

        return {"public_key": pubkey} if pubkey else None

    # ----- formatting / helpers -----

    def build_conversation(self, channel_idx: Optional[int], user_text: str) -> List[Tuple[str, str]]:
        if channel_idx is None:
            return list(self.dm_history) + [("user", user_text)]
        return list(self.history[channel_idx]) + [("user", user_text)]

    def format_chan_reply(self, sender: str, msg: str) -> str:
        msg = (msg or "").strip()
        return f"@[{sender}] {msg}" if sender and msg else msg

    def get_help_string(self, is_ai: bool, is_weather: bool) -> str:
        cmds = [self.help_trigger, self.ping_trigger]
        if is_weather:
            cmds.append(f"{self.weather_trigger} [loc]")
        if is_ai:
            cmds.append(f"{self.trigger} [msg]")
        return "Commands: " + ", ".join(cmds)

    def get_telemetry_string(self, payload: Dict[str, Any]) -> str:
        snr = payload.get("SNR") or payload.get("rxSnr") or payload.get("rx_snr") or payload.get("snr")
        rssi = payload.get("RSSI") or payload.get("rxRssi") or payload.get("rx_rssi") or payload.get("rssi")
        path_len = payload.get("path_len")

        hops = None
        if path_len is not None:
            hops = path_len
        else:
            hop_limit = payload.get("hopLimit") or payload.get("hop_limit")
            hop_start = payload.get("hopStart") or payload.get("hop_start")
            if hop_limit is not None:
                try:
                    hl = int(hop_limit)
                    hs = int(hop_start) if hop_start is not None else hl
                    hops = (hs - hl) if hs >= hl else 0
                except (TypeError, ValueError):
                    pass

        try:
            return self.ping_template.format(
                snr=snr if snr is not None else "?",
                rssi=rssi if rssi is not None else "?",
                hops=hops if hops is not None else "?",
            )
        except Exception:
            return f"pong [SNR: {snr if snr is not None else '?'}, RSSI: {rssi if rssi is not None else '?'}dBm, Hops: {hops if hops is not None else '?'}]"

    @staticmethod
    def split_sender_and_body(text: str) -> Tuple[str, str]:
        t = (text or "").strip()
        if ": " in t:
            name, body = t.split(": ", 1)
            if name and len(name.strip()) <= 40:
                return name.strip(), body.strip()
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
            for k, t0 in list(self._seen.items()):
                if now - t0 > self.dedupe_window_s:
                    self._seen.pop(k, None)
            if key in self._seen:
                if self.debug:
                    print(f"[DBG] duplicate dropped key={key}")
                return True
            self._seen[key] = now
            return False

    async def send_channel_text(self, ch_idx: int, sender: str, text: str) -> None:
        parts = chunk_text(text, self.max_reply_chars)
        for i, part in enumerate(parts, start=1):
            msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
            out = self.format_chan_reply(sender, msg)
            if out:
                await self.mesh.commands.send_chan_msg(ch_idx, out)

    async def send_dm_text(self, dst: Dict[str, Any], text: str) -> None:
        parts = chunk_text(text, self.max_reply_chars)
        for i, part in enumerate(parts, start=1):
            msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
            if msg:
                await self.mesh.commands.send_msg(dst, msg)

    # ----- weather -----

    async def fetch_weather(self, location: str) -> str:
        if not location:
            return "No weather location specified."

        loc = urllib.parse.quote(location)
        url = f"https://wttr.in/{loc}?format=j1"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            current = data.get("current_condition", [{}])[0]
            today = data.get("weather", [{}])[0]

            cond = current.get("weatherDesc", [{"value": "Unknown"}])[0].get("value", "Unknown")
            temp_f = current.get("temp_F", "?")
            temp_c = current.get("temp_C", "?")
            wind_mph = current.get("windspeedMiles", "?")
            wind_kmh = current.get("windspeedKmph", "?")
            hum = current.get("humidity", "?")

            high_f = today.get("maxtempF", "?")
            low_f = today.get("mintempF", "?")
            high_c = today.get("maxtempC", "?")
            low_c = today.get("mintempC", "?")

            rain = max([int(h.get("chanceofrain", "0")) for h in today.get("hourly", [])] or [0])
            snow = max([int(h.get("chanceofsnow", "0")) for h in today.get("hourly", [])] or [0])

            display_loc = location.title()
            if self.weather_units == "C":
                msg = (
                    f"🌤️ {display_loc}: {cond} {temp_c}°C "
                    f"(Wind {wind_kmh}km/h, Hum {hum}%). "
                    f"Today: High {high_c}°C / Low {low_c}°C"
                )
            else:
                msg = (
                    f"🌤️ {display_loc}: {cond} {temp_f}°F "
                    f"(Wind {wind_mph}mph, Hum {hum}%). "
                    f"Today: High {high_f}°F / Low {low_f}°F"
                )

            if snow > 0:
                msg += f", Snow {snow}%"
            elif rain > 0:
                msg += f", Rain {rain}%"

            return msg

        except Exception as e:
            if self.debug:
                print(f"[DBG] Weather fetch error for {location}: {e}")
            return f"Could not find weather for '{location}' (API error)."

    async def scheduled_weather_loop(self, stop_event: asyncio.Event) -> None:
        if not self.weather_times or not self.weather_location or not self.weather_channels:
            return

        parsed_times: List[Tuple[int, int]] = []
        for t in self.weather_times:
            hhmm = t.split(":")
            if len(hhmm) == 2 and hhmm[0].isdigit() and hhmm[1].isdigit():
                parsed_times.append((int(hhmm[0]), int(hhmm[1])))

        if not parsed_times:
            return

        last_sent_min = -1
        while not stop_event.is_set():
            now = time.localtime()
            current = (now.tm_hour, now.tm_min)
            if current in parsed_times and now.tm_min != last_sent_min:
                last_sent_min = now.tm_min
                if self.debug:
                    print(f"[DBG] Triggering scheduled weather for {current}")
                ans = await self.fetch_weather(self.weather_location)
                async with self._llm_lock:
                    for ch_idx in self.weather_channels:
                        await self.send_channel_text(ch_idx, "", ans)
                        await asyncio.sleep(2)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass

    async def weather_alerts_loop(self, zones: str, interval_m: float, stop_event: asyncio.Event) -> None:
        if not zones or interval_m <= 0 or not self.weather_channels:
            return

        url = f"https://api.weather.gov/alerts/active?zone={urllib.parse.quote(zones)}"
        headers = {"User-Agent": "MeshCore-Weather-Alert-Bot/1.0 (https://github.com/)"}
        interval_s = interval_m * 60.0

        while not stop_event.is_set():
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                for feature in data.get("features", []):
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

                    async with self._llm_lock:
                        for ch_idx in self.weather_channels:
                            await self.send_channel_text(ch_idx, "", msg)
                            await asyncio.sleep(2)

            except Exception as e:
                if self.debug:
                    print(f"[DBG] Weather alert polling error: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass

    # ----- event handlers -----

    async def on_channel_msg(self, ev) -> None:
        self.last_rx_time = time.time()

        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        ch_idx = p.get("channel_idx")
        is_ai_chan = ch_idx in self.ai_channels
        is_weather_chan = ch_idx in self.weather_channels
        if not is_ai_chan and not is_weather_chan:
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

        body_lower = body.lower()

        if body_lower == self.help_trigger.lower() or body_lower.startswith(self.help_trigger.lower() + " "):
            await self.send_channel_text(ch_idx, sender, self.get_help_string(is_ai_chan, is_weather_chan))
            return

        if body_lower == self.ping_trigger.lower() or body_lower.startswith(self.ping_trigger.lower() + " "):
            await self.send_channel_text(ch_idx, sender, self.get_telemetry_string(p))
            return

        if is_weather_chan:
            w = self.weather_trigger.lower()
            if body_lower == w or body_lower.startswith(w + " "):
                req_loc = body[len(self.weather_trigger):].strip()
                target_loc = req_loc if req_loc else self.weather_location
                ans = await self.fetch_weather(target_loc)
                async with self._llm_lock:
                    await self.send_channel_text(ch_idx, sender, ans)
                return

        if is_ai_chan:
            user = self.extract_after_trigger(body)
            if not user:
                return

            async with self._llm_lock:
                self.history[ch_idx].append(("user", user))
                conversation = self.build_conversation(ch_idx, user)
                try:
                    answer = await self.llm.generate(self.system_prompt, conversation)
                except Exception as e:
                    answer = f"LLM error: {e}"
                self.history[ch_idx].append(("assistant", answer))
                await self.send_channel_text(ch_idx, sender, answer)

    async def on_dm_msg(self, ev) -> None:
        self.last_rx_time = time.time()

        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        text = p.get("text")
        if not isinstance(text, str):
            return

        _sender, body = self.split_sender_and_body(text)
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

        body_lower = body.lower()

        if body_lower == self.help_trigger.lower() or body_lower.startswith(self.help_trigger.lower() + " "):
            await self.send_dm_text(dst, self.get_help_string(is_ai=True, is_weather=True))
            return

        if body_lower == self.ping_trigger.lower() or body_lower.startswith(self.ping_trigger.lower() + " "):
            await self.send_dm_text(dst, self.get_telemetry_string(p))
            return

        w = self.weather_trigger.lower()
        if body_lower == w or body_lower.startswith(w + " "):
            req_loc = body[len(self.weather_trigger):].strip()
            target_loc = req_loc if req_loc else self.weather_location
            ans = await self.fetch_weather(target_loc)
            async with self._llm_lock:
                await self.send_dm_text(dst, ans)
            return

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            self.dm_history.append(("user", user))
            conversation = self.build_conversation(None, user)
            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"
            self.dm_history.append(("assistant", answer))
            await self.send_dm_text(dst, answer)


# ---------------------------
# session / reconnect
# ---------------------------

async def run_bot_once(llm: LLMClient) -> None:
    ai_targets = [c.strip() for c in env_str("MESHCORE_AI_CHANNELS", "#avl-ai").split(",") if c.strip()]
    weather_targets = [c.strip() for c in env_str("MESHCORE_WEATHER_CHANNELS", "#weather-avl").split(",") if c.strip()]
    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    print("[INFO] connecting to MeshCore...")
    mesh = await create_mesh_connection()
    await asyncio.sleep(1.0)
    await mesh.start_auto_message_fetching()

    ai_channel_map = await resolve_channels(mesh, ai_targets, scan_max)
    weather_channel_map = await resolve_channels(mesh, weather_targets, scan_max)

    print("[OK] AI Channel map:")
    for idx, name in ai_channel_map.items():
        print(f"  idx={idx} -> {name}")
    if not ai_channel_map:
        print("  (None configured)")

    print("[OK] Weather Channel map:")
    for idx, name in weather_channel_map.items():
        print(f"  idx={idx} -> {name}")
    if not weather_channel_map:
        print("  (None configured)")

    bot = MeshBot(mesh, llm, ai_channel_map, weather_channel_map)

    mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
    mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

    await bot.refresh_contacts_best_effort()

    stop_event = asyncio.Event()
    bg_tasks: List[asyncio.Task] = []

    if bot.weather_times and bot.weather_location and weather_channel_map:
        bg_tasks.append(asyncio.create_task(bot.scheduled_weather_loop(stop_event)))
        print(f"[OK] Scheduled weather enabled: Location='{bot.weather_location}', Times={bot.weather_times}, Units={bot.weather_units}")

    alerts_zones = env_str("WEATHER_ALERTS_NWS_ZONES", "")
    alerts_interval_m = env_float("WEATHER_ALERTS_POLL_INTERVAL_M", 15.0)
    if alerts_zones and alerts_interval_m > 0 and weather_channel_map:
        bg_tasks.append(asyncio.create_task(bot.weather_alerts_loop(alerts_zones, alerts_interval_m, stop_event)))
        print(f"[OK] NWS weather alerts polling enabled: Zones='{alerts_zones}', Interval={alerts_interval_m}m")

    print("\n[OK] Connected and Listening.")
    print(f"[TEST] Help      (Any channel or DM):  '{bot.help_trigger}'")
    print(f"[TEST] Ping      (Any channel or DM):  '{bot.ping_trigger}'")
    print(f"[TEST] AI Query  (AI Channels or DM):  '{bot.trigger} hello'")
    print(f"[TEST] Weather   (Weather Ch. or DM):  '{bot.weather_trigger}' or '{bot.weather_trigger} Paris'\n")

    loop = asyncio.get_running_loop()
    disconnect_future: asyncio.Future[str] = loop.create_future()

    async def on_disconnected(_ev) -> None:
        print("[WARN] MeshCore disconnected event received")
        if not disconnect_future.done():
            disconnect_future.set_result("disconnected_event")

    mesh.subscribe(EventType.DISCONNECTED, on_disconnected)

    health_interval_s = env_int("HEALTHCHECK_INTERVAL_S", 10)
    health_idle_timeout_s = env_int("HEALTH_IDLE_TIMEOUT_S", 300)

    async def health_monitor() -> None:
        while not disconnect_future.done():
            idle = time.time() - bot.last_rx_time
            if idle > health_idle_timeout_s:
                print(f"[WARN] no packets received for {int(idle)}s -> forcing reconnect")
                if not disconnect_future.done():
                    disconnect_future.set_result("rx_timeout")
                return

            try:
                await asyncio.wait_for(disconnect_future, timeout=health_interval_s)
            except asyncio.TimeoutError:
                pass

    health_task = asyncio.create_task(health_monitor())

    reason = "unknown"
    try:
        reason = await disconnect_future
    except asyncio.CancelledError:
        if bot.debug:
            print("[DBG] run_bot_once cancelled")
    finally:
        stop_event.set()

        for task in bg_tasks:
            task.cancel()
        health_task.cancel()

        for task in bg_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if bot.debug:
                    print(f"[DBG] background task ended with error: {e}")

        try:
            await health_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if bot.debug:
                print(f"[DBG] health task ended with error: {e}")

        print(f"[INFO] Closing MeshCore connection... reason={reason}")
        try:
            await close_mesh(mesh)
        except Exception as e:
            if bot.debug:
                print(f"[DBG] error while closing MeshCore connection: {e}")


async def main() -> None:
    llm = build_llm()

    reconnect_delay = env_int("RECONNECT_DELAY_S", 5)
    reconnect_max_delay = env_int("RECONNECT_MAX_DELAY_S", 60)
    delay = reconnect_delay

    try:
        while True:
            try:
                await run_bot_once(llm)
                delay = reconnect_delay
            except asyncio.CancelledError:
                print("[INFO] main task cancelled; shutting down cleanly")
                break
            except Exception as e:
                print(f"[WARN] bot session ended unexpectedly: {e}")

            print(f"[INFO] reconnecting in {delay}s...")
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                print("[INFO] sleep cancelled; shutting down cleanly")
                break

            delay = min(delay * 2, reconnect_max_delay)
    finally:
        try:
            await llm.aclose()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user.")
    except asyncio.CancelledError:
        print("\n[INFO] Bot cancelled; exiting cleanly.")
