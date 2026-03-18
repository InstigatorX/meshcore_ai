#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ separates AI channels and Weather channels.
+ Scheduled daily weather broadcasts and on-demand `!weather [location]` fetching.
+ Includes today's forecast (High/Low/Rain chance)
+ Configurable polling for US National Weather Service (NWS) severe weather alerts.
+ Global `!ping` command for testing connectivity (returns SNR, RSSI, and Hops).
+ Global `!help` command listing available triggers based on context.
+ Watchdog loop: Detects dead links by idle time since last received MeshCore event.

DM replies require a destination with full 'public_key'. We resolve DM sender via
pubkey_prefix by caching contacts. DMs support all commands.

Safe fixes applied:
- Prevent duplicate latest user turn from being sent to the LLM
- Fix scheduled weather duplicate/minute logic across different hours
- Allow AI/weather channel groups to be optional if not found
- Make dedupe key safer when sender_timestamp is missing
- Validate weather units
- Add clean shutdown for HTTP clients / mesh if process exits normally

Telemetry fix applied:
- Remove unsupported/undocumented mesh.events harvester
- Use EventType.RX_LOG_DATA as telemetry source
- Keep a small rolling RX log buffer
- Match !ping replies against recent RX_LOG_DATA entries by:
  - expected message type (GRP_TXT for channels, TEXT_MSG for DMs)
  - matching path_len
  - closest local process arrival time
- Retry telemetry match once after a short delay to avoid event-order races

Minor tweaks applied:
- Preserve uppercase location names like AVL
- Render 0 hops as "Direct"

Watchdog fixes applied:
- No more fragile get_channel(0) liveness probe
- No more os._exit(1) hard exit
- Watchdog now tracks idle time since last received MeshCore event
"""

import asyncio
import os
import re
import sys
import time
import urllib.parse
from collections import deque
from contextlib import suppress
from typing import Any, Deque, Dict, List, Optional, Tuple, Set

import httpx
from meshcore import MeshCore, EventType

# Gemini optional
try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # noqa: N816


class WatchdogTimeout(RuntimeError):
    pass


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


async def resolve_channels(mesh: MeshCore, channel_names: List[str], max_channels: int = 16) -> Dict[int, str]:
    if not channel_names:
        return {}

    want = {normalize_channel_name(c) for c in channel_names if c.strip()}
    if not want:
        return {}

    found: Dict[int, str] = {}

    for idx in range(max_channels):
        try:
            ev = await mesh.commands.get_channel(idx)
        except Exception:
            continue

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

    async def aclose(self) -> None:
        return None


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
        ai_channels: Dict[int, str],
        weather_channels: Dict[int, str],
        trigger: str,
        ping_trigger: str,
        help_trigger: str,
        ping_template: str,
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
        self.ai_channels = ai_channels
        self.weather_channels = weather_channels
        self.trigger = trigger.strip()
        self.ping_trigger = ping_trigger.strip()
        self.help_trigger = help_trigger.strip()
        self.ping_template = ping_template
        self.max_reply_chars = max_reply_chars
        self.debug = debug
        self.dedupe_window_s = dedupe_window_s
        self.system_prompt = system_prompt

        self.weather_location = weather_location
        self.weather_times = weather_times
        self.weather_trigger = weather_trigger.strip()

        units = weather_units.upper().strip()
        if units not in ("F", "C"):
            if self.debug:
                print(f"[DBG] Invalid WEATHER_UNITS={weather_units!r}; defaulting to 'F'")
            units = "F"
        self.weather_units = units

        self._seen_alerts: Set[str] = set()

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(self.trigger)}(\s+|$)", re.IGNORECASE)

        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            idx: deque(maxlen=history_turns * 2) for idx in ai_channels.keys()
        }

        self._llm_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

        self._dedupe_lock = asyncio.Lock()
        self._seen_ts: Dict[Tuple[str, int, str, int, str], float] = {}

        self._contacts_lock = asyncio.Lock()
        self._contacts_by_pubkey: Dict[str, Dict[str, Any]] = {}
        self._contacts_by_prefix: Dict[str, str] = {}

        self._rxlog_lock = asyncio.Lock()
        self._recent_rxlog: Deque[Dict[str, Any]] = deque(maxlen=100)

        self.last_activity_ts = time.monotonic()

    def mark_activity(self) -> None:
        self.last_activity_ts = time.monotonic()

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
            print(f"[DBG] cached contact pubkey_prefix={prefix} name={name}")

    async def on_contacts_event(self, ev) -> None:
        self.mark_activity()

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

    # ---------------- RX log telemetry matching ----------------

    async def on_rx_log_data(self, ev) -> None:
        self.mark_activity()

        payload = ev.payload or {}
        attrs = getattr(ev, "attributes", None) or {}

        if not isinstance(payload, dict):
            payload = {}
        if not isinstance(attrs, dict):
            attrs = {}

        def pick(*keys):
            for k in keys:
                if k in payload and payload.get(k) is not None:
                    return payload.get(k)
                if k in attrs and attrs.get(k) is not None:
                    return attrs.get(k)
            return None

        entry = {
            "local_ts": time.monotonic(),
            "recv_time": pick("recv_time"),
            "snr": pick("snr", "rx_snr", "SNR", "rxSnr"),
            "rssi": pick("rssi", "rx_rssi", "RSSI", "rxRssi"),
            "path_len": pick("path_len"),
            "payload_typename": pick("payload_typename"),
            "pkt_hash": pick("pkt_hash"),
        }

        async with self._rxlog_lock:
            self._recent_rxlog.append(entry)

        if self.debug:
            print(
                "[DBG] RX_LOG cached "
                f"type={entry['payload_typename']} "
                f"path_len={entry['path_len']} "
                f"snr={entry['snr']} rssi={entry['rssi']} "
                f"local_ts={entry['local_ts']:.3f}"
            )

    async def match_telemetry_for_message(
        self,
        is_dm: bool,
        msg_payload: Dict[str, Any],
        msg_local_ts: float,
    ) -> Dict[str, Any]:
        expected_type = "TEXT_MSG" if is_dm else "GRP_TXT"
        msg_path_len = msg_payload.get("path_len")

        async with self._rxlog_lock:
            candidates = list(self._recent_rxlog)

        candidates = [
            c for c in candidates
            if isinstance(c.get("local_ts"), (int, float))
            and abs(float(c["local_ts"]) - msg_local_ts) <= 2.0
        ]

        scored: List[Tuple[int, float, Dict[str, Any]]] = []

        for c in candidates:
            if c.get("snr") is None and c.get("rssi") is None:
                continue

            score = 0

            rx_type = c.get("payload_typename")
            if rx_type is not None:
                if rx_type == expected_type:
                    score += 60
                else:
                    score -= 40

            rx_path_len = c.get("path_len")
            if msg_path_len is not None and rx_path_len is not None:
                try:
                    if int(rx_path_len) == int(msg_path_len):
                        score += 40
                    else:
                        score -= abs(int(rx_path_len) - int(msg_path_len)) * 10
                except (ValueError, TypeError):
                    pass

            dt_local = abs(float(c["local_ts"]) - msg_local_ts)
            score += max(0, int((2.0 - dt_local) * 100))

            scored.append((score, -dt_local, c))

        if not scored:
            if self.debug:
                print(
                    "[DBG] no telemetry match "
                    f"expected_type={expected_type} msg_path_len={msg_path_len} "
                    f"msg_local_ts={msg_local_ts:.3f}"
                )
            return {}

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][2]

        if self.debug:
            print(
                "[DBG] matched telemetry "
                f"expected_type={expected_type} msg_path_len={msg_path_len} "
                f"snr={best.get('snr')} rssi={best.get('rssi')} "
                f"rx_path_len={best.get('path_len')} "
                f"rx_type={best.get('payload_typename')} "
                f"rx_local_ts={best.get('local_ts'):.3f} "
                f"msg_local_ts={msg_local_ts:.3f}"
            )

        return best

    # ---------------- Weather & Alerts Logic ----------------

    async def fetch_weather(self, location: str) -> str:
        if not location:
            return "No weather location specified."
    
        loc_encoded = urllib.parse.quote(location)
        url = f"https://wttr.in/{loc_encoded}?format=j1"

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "meshcore-llm-bot/1.0"}
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                raw_text = resp.text

                if self.debug:
                    print(f"[DBG] wttr status={resp.status_code} content-type={content_type}")
                    print(f"[DBG] wttr first 300 chars: {raw_text[:300]!r}")

                data = resp.json()

                # wttr sometimes wraps payload under top-level "data"
                if isinstance(data, dict) and isinstance(data.get("data"), dict):
                    data = data["data"]

                current_list = data.get("current_condition")
                weather_list = data.get("weather")

            if not isinstance(current_list, list) or not current_list:
                if self.debug:
                    print(f"[DBG] wttr missing current_condition. keys={list(data.keys())}")
                return f"Weather unavailable for '{location}' right now."

            if not isinstance(weather_list, list) or not weather_list:
                if self.debug:
                    print(f"[DBG] wttr missing weather forecast. keys={list(data.keys())}")
                return f"Forecast unavailable for '{location}' right now."

            current = current_list[0] if isinstance(current_list[0], dict) else {}
            today = weather_list[0] if isinstance(weather_list[0], dict) else {}

            weather_desc_list = current.get("weatherDesc")
            if isinstance(weather_desc_list, list) and weather_desc_list and isinstance(weather_desc_list[0], dict):
                cond = weather_desc_list[0].get("value", "Unknown")
            else:
                cond = "Unknown"

            temp_f = current.get("temp_F", "?")
            temp_c = current.get("temp_C", "?")
            wind_mph = current.get("windspeedMiles", "?")
            wind_kmh = current.get("windspeedKmph", "?")
            hum = current.get("humidity", "?")

            high_f = today.get("maxtempF", "?")
            low_f = today.get("mintempF", "?")
            high_c = today.get("maxtempC", "?")
            low_c = today.get("mintempC", "?")
    
            hourly = today.get("hourly", [])
            rain_chances = []
            snow_chances = []
            if isinstance(hourly, list):
                for h in hourly:
                    if not isinstance(h, dict):
                        continue
                    try:
                        rain_chances.append(int(h.get("chanceofrain", "0")))
                    except Exception:
                        pass
                    try:
                        snow_chances.append(int(h.get("chanceofsnow", "0")))
                    except Exception:
                        pass

            max_rain = max(rain_chances) if rain_chances else 0
            max_snow = max(snow_chances) if snow_chances else 0

            display_loc = location.strip()
            if not display_loc.isupper():
                display_loc = display_loc.title()

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

            if max_snow > 0:
                msg += f", Snow {max_snow}%"
            elif max_rain > 0:
                msg += f", Rain {max_rain}%"

            return msg

        except Exception as e:
            if self.debug:
                print(f"[DBG] Weather fetch error for {location}: {e}")
            return f"Could not find weather for '{location}' (API Error)."

    async def send_channel_text(self, ch_idx: int, text: str, sender: str = "") -> None:
        parts = chunk_text(text, self.max_reply_chars)
        async with self._send_lock:
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_chan_reply(sender, msg)
                if out:
                    await self.mesh.commands.send_chan_msg(ch_idx, out)
                    await asyncio.sleep(2)

    async def send_dm_text(self, dst: Dict[str, Any], text: str) -> None:
        parts = chunk_text(text, self.max_reply_chars)
        async with self._send_lock:
            for i, part in enumerate(parts, start=1):
                msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                out = self.format_dm_reply(msg)
                if out:
                    await self.mesh.commands.send_msg(dst, out)
                    await asyncio.sleep(2)

    async def scheduled_weather_loop(self) -> None:
        if not self.weather_times or not self.weather_location or not self.weather_channels:
            return

        parsed_times: List[Tuple[int, int]] = []
        for t_str in self.weather_times:
            parts = t_str.split(":")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                hh = int(parts[0])
                mm = int(parts[1])
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    parsed_times.append((hh, mm))

        if not parsed_times:
            if self.debug:
                print("[DBG] No valid WEATHER_SCHEDULED_TIMES parsed.")
            return

        last_sent_key: Optional[Tuple[int, int, int, int]] = None

        while True:
            now = time.localtime()
            current_time = (now.tm_hour, now.tm_min)
            send_key = (now.tm_year, now.tm_yday, now.tm_hour, now.tm_min)

            if current_time in parsed_times and send_key != last_sent_key:
                last_sent_key = send_key

                if self.debug:
                    print(f"[DBG] Triggering scheduled weather for {current_time}")

                ans = await self.fetch_weather(self.weather_location)

                for ch_idx in self.weather_channels.keys():
                    await self.send_channel_text(ch_idx, ans)

            await asyncio.sleep(10)

    async def weather_alerts_loop(self, zones: str, interval_m: float) -> None:
        if not zones or interval_m <= 0 or not self.weather_channels:
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

                    for ch_idx in self.weather_channels.keys():
                        await self.send_channel_text(ch_idx, msg)

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
        return b[idx + len(self.trigger):].strip(" \t:,-")

    def get_sender_identity(self, payload: Dict[str, Any], sender: str) -> str:
        for key in ("public_key", "pubkey_prefix", "sender", "sender_id", "from"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
            if isinstance(v, int):
                return str(v)
        return (sender or "").strip().lower()

    async def dedupe_drop(self, scope: str, ch_idx: int, sender_id: str, sender_ts: int, body: str) -> bool:
        key = (scope, ch_idx, sender_id, sender_ts, body)
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

    def get_help_string(self, is_ai: bool, is_weather: bool) -> str:
        cmds = [self.help_trigger, self.ping_trigger]
        if is_weather:
            cmds.append(f"{self.weather_trigger} [loc]")
        if is_ai:
            cmds.append(f"{self.trigger} [msg]")
        return "Commands: " + ", ".join(cmds)

    def get_telemetry_string(self, p: Dict[str, Any]) -> str:
        snr = p.get("SNR") or p.get("snr") or p.get("rxSnr") or p.get("rx_snr")
        rssi = p.get("RSSI") or p.get("rssi") or p.get("rxRssi") or p.get("rx_rssi")

        path_len = p.get("path_len")
        hops = None
        if path_len is not None:
            hops = path_len
        else:
            hl = p.get("hopLimit") or p.get("hop_limit")
            hs = p.get("hopStart") or p.get("hop_start")
            if hl is not None:
                try:
                    hl = int(hl)
                    hs = int(hs) if hs is not None else hl
                    hops = (hs - hl) if hs >= hl else 0
                except (ValueError, TypeError):
                    pass

        safe_snr = snr if snr is not None else "?"
        safe_rssi = rssi if rssi is not None else "?"
        if hops is None:
            safe_hops = "?"
        elif hops == 0:
            safe_hops = "Direct"
        else:
            safe_hops = hops

        try:
            return self.ping_template.format(snr=safe_snr, rssi=safe_rssi, hops=safe_hops)
        except Exception as e:
            if self.debug:
                print(f"[DBG] PING_TEMPLATE formatting error: {e}")
            return f"pong [SNR: {safe_snr}, RSSI: {safe_rssi}dBm, Hops: {safe_hops}]"

    # ---------------- Event handlers ----------------

    async def on_channel_msg(self, ev) -> None:
        self.mark_activity()

        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        msg_local_ts = time.monotonic()

        ch_idx = p.get("channel_idx")
        if not isinstance(ch_idx, int):
            return

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

        sender_id = self.get_sender_identity(p, sender)

        if await self.dedupe_drop("chan", ch_idx, sender_id, sender_ts, body):
            return

        body_lower = body.lower()

        h_trigger = self.help_trigger.lower()
        if body_lower == h_trigger or body_lower.startswith(h_trigger + " "):
            reply_text = self.get_help_string(is_ai_chan, is_weather_chan)
            await self.send_channel_text(ch_idx, reply_text, sender)
            return

        p_trigger = self.ping_trigger.lower()
        if body_lower == p_trigger or body_lower.startswith(p_trigger + " "):
            matched = await self.match_telemetry_for_message(False, p, msg_local_ts)
            if not matched:
                await asyncio.sleep(0.22)
                matched = await self.match_telemetry_for_message(False, p, time.monotonic())

            merged = dict(p)
            merged.update({k: v for k, v in matched.items() if v is not None})
            reply_text = self.get_telemetry_string(merged)
            await self.send_channel_text(ch_idx, reply_text, sender)
            return

        if is_weather_chan:
            w_trigger = self.weather_trigger.lower()
            if body_lower == w_trigger or body_lower.startswith(w_trigger + " "):
                req_loc = body[len(self.weather_trigger):].strip()
                target_loc = req_loc if req_loc else self.weather_location
                ans = await self.fetch_weather(target_loc)
                await self.send_channel_text(ch_idx, ans, sender)
                return

        if is_ai_chan:
            user = self.extract_after_trigger(body)
            if not user:
                return

            async with self._llm_lock:
                conversation = self.build_conversation(ch_idx, user)

                try:
                    answer = await self.llm.generate(self.system_prompt, conversation)
                except Exception as e:
                    answer = f"LLM error: {e}"

                self.history[ch_idx].append(("user", user))
                self.history[ch_idx].append(("assistant", answer))

            await self.send_channel_text(ch_idx, answer, sender)

    async def on_dm_msg(self, ev) -> None:
        self.mark_activity()

        p = ev.payload or {}
        if not isinstance(p, dict):
            return

        msg_local_ts = time.monotonic()

        text = p.get("text")
        if not isinstance(text, str):
            return

        sender, body = self.split_sender_and_body(text)
        sender_ts = p.get("sender_timestamp")
        if not isinstance(sender_ts, int):
            sender_ts = -1

        sender_id = self.get_sender_identity(p, sender)

        if await self.dedupe_drop("dm", -1, sender_id, sender_ts, body):
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

        h_trigger = self.help_trigger.lower()
        if body_lower == h_trigger or body_lower.startswith(h_trigger + " "):
            reply_text = self.get_help_string(is_ai=True, is_weather=True)
            await self.send_dm_text(dst, reply_text)
            return

        p_trigger = self.ping_trigger.lower()
        if body_lower == p_trigger or body_lower.startswith(p_trigger + " "):
            matched = await self.match_telemetry_for_message(True, p, msg_local_ts)
            if not matched:
                await asyncio.sleep(0.22)
                matched = await self.match_telemetry_for_message(True, p, time.monotonic())

            merged = dict(p)
            merged.update({k: v for k, v in matched.items() if v is not None})
            reply_text = self.get_telemetry_string(merged)
            await self.send_dm_text(dst, reply_text)
            return

        w_trigger = self.weather_trigger.lower()
        if body_lower == w_trigger or body_lower.startswith(w_trigger + " "):
            req_loc = body[len(self.weather_trigger):].strip()
            target_loc = req_loc if req_loc else self.weather_location
            ans = await self.fetch_weather(target_loc)
            await self.send_dm_text(dst, ans)
            return

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            dummy_idx = 999
            if dummy_idx not in self.history:
                ml = self.history[list(self.ai_channels.keys())[0]].maxlen if self.ai_channels else 12
                self.history[dummy_idx] = deque(maxlen=ml)

            conversation = self.build_conversation(dummy_idx, user)

            try:
                answer = await self.llm.generate(self.system_prompt, conversation)
            except Exception as e:
                answer = f"LLM error: {e}"

            self.history[dummy_idx].append(("user", user))
            self.history[dummy_idx].append(("assistant", answer))

        await self.send_dm_text(dst, answer)


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


async def watchdog_loop(bot: ChannelLLMBot, debug: bool) -> None:
    idle_timeout_s = env_int("WATCHDOG_IDLE_TIMEOUT_S", 300)
    check_interval_s = env_int("WATCHDOG_CHECK_INTERVAL_S", 30)

    while True:
        await asyncio.sleep(check_interval_s)
        idle = time.monotonic() - bot.last_activity_ts

        if debug:
            print(f"[DBG] Watchdog idle={idle:.1f}s timeout={idle_timeout_s}s")

        if idle > idle_timeout_s:
            raise WatchdogTimeout(
                f"No MeshCore events received for {int(idle)}s (timeout={idle_timeout_s}s)"
            )


async def safe_close_mesh(mesh: Any, debug: bool) -> None:
    for meth in ("close", "disconnect", "aclose", "stop"):
        fn = getattr(mesh, meth, None)
        if fn is None:
            continue
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            if debug:
                print(f"[DBG] Mesh closed via {meth}()")
            return
        except Exception as e:
            if debug:
                print(f"[DBG] Mesh close method {meth} failed: {e}")


async def main() -> None:
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")
    print(f"[INFO] Booting. Debug mode is: {'ENABLED' if debug else 'DISABLED'}")

    ai_channels_raw = env_str("MESHCORE_AI_CHANNELS", "#avl-ai")
    target_ai_channels = [c.strip() for c in ai_channels_raw.split(",") if c.strip()]

    weather_channels_raw = env_str("MESHCORE_WEATHER_CHANNELS", "#weather-avl")
    target_weather_channels = [c.strip() for c in weather_channels_raw.split(",") if c.strip()]

    scan_max = env_int("CHANNEL_SCAN_MAX", 16)

    trigger = env_str("AI_TRIGGER", "!ai").strip()
    ping_trigger = env_str("PING_TRIGGER", "!ping").strip()
    help_trigger = env_str("HELP_TRIGGER", "!help").strip()
    ping_template = env_str("PING_TEMPLATE", "pong [SNR: {snr}, RSSI: {rssi}dBm, Hops: {hops}]")

    max_reply_chars = env_int("MAX_REPLY_CHARS", 180)
    history_turns = env_int("HISTORY_TURNS", 6)
    dedupe_window_s = env_float("DEDUPE_WINDOW_S", 3.0)
    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    weather_location = env_str("WEATHER_LOCATION", "")
    weather_times_raw = env_str("WEATHER_SCHEDULED_TIMES", "")
    weather_times = [t.strip() for t in weather_times_raw.split(",") if t.strip()]
    weather_trigger = env_str("WEATHER_TRIGGER", "!weather").strip()
    weather_units = env_str("WEATHER_UNITS", "F")

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
    tasks: List[asyncio.Task[Any]] = []

    try:
        await mesh.start_auto_message_fetching()

        ai_channel_map = await resolve_channels(mesh, target_ai_channels, max_channels=scan_max)
        weather_channel_map = await resolve_channels(mesh, target_weather_channels, max_channels=scan_max)

        print("[OK] AI Channel map:")
        if not ai_channel_map:
            print("  (None configured/found)")
        for idx, name in ai_channel_map.items():
            print(f"  idx={idx} -> {name}")

        print("[OK] Weather Channel map:")
        if not weather_channel_map:
            print("  (None configured/found)")
        for idx, name in weather_channel_map.items():
            print(f"  idx={idx} -> {name}")

        bot = ChannelLLMBot(
            mesh=mesh,
            llm=llm,
            ai_channels=ai_channel_map,
            weather_channels=weather_channel_map,
            trigger=trigger,
            ping_trigger=ping_trigger,
            help_trigger=help_trigger,
            ping_template=ping_template,
            max_reply_chars=max_reply_chars,
            history_turns=history_turns,
            dedupe_window_s=dedupe_window_s,
            debug=debug,
            system_prompt=system_prompt,
            weather_location=weather_location,
            weather_times=weather_times,
            weather_trigger=weather_trigger,
            weather_units=weather_units,
        )

        mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
        mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
        mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)

        mesh.subscribe(EventType.RX_LOG_DATA, bot.on_rx_log_data)
        mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
        mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

        await bot.refresh_contacts_best_effort()

        if weather_times and weather_location and weather_channel_map:
            tasks.append(asyncio.create_task(bot.scheduled_weather_loop(), name="scheduled_weather"))
            print(f"[OK] Scheduled weather enabled: Location='{weather_location}', Times={weather_times}, Units={bot.weather_units}")

        if alerts_zones and alerts_interval_m > 0 and weather_channel_map:
            tasks.append(asyncio.create_task(bot.weather_alerts_loop(alerts_zones, alerts_interval_m), name="weather_alerts"))
            print(f"[OK] NWS weather alerts polling enabled: Zones='{alerts_zones}', Interval={alerts_interval_m}m")

        tasks.append(asyncio.create_task(watchdog_loop(bot, debug), name="watchdog"))

        print(f"\n[OK] Connected and Listening.")
        print(f"[LLM] Backend={backend}")
        print("--- COMMANDS ---")
        print(f" [TEST] Help      (Any channel or DM):  '{help_trigger}'")
        print(f" [TEST] Ping      (Any channel or DM):  '{ping_trigger}'")
        print(f" [TEST] AI Query  (AI Channels or DM):  '{trigger} hello'")
        print(f" [TEST] Weather   (Weather Ch. or DM):  '{weather_trigger}' or '{weather_trigger} Paris'\n")

        await asyncio.sleep(float("inf"))

    finally:
        task_errors: List[BaseException] = []

        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                task_errors.append(e)

        with suppress(Exception):
            await llm.aclose()

        with suppress(Exception):
            await safe_close_mesh(mesh, debug)

        if task_errors:
            raise task_errors[0]


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted, shutting down.")
        sys.exit(0)
    except WatchdogTimeout as e:
        print(f"\n[ERR] Watchdog timeout: {e}")
        print("[INFO] Exiting cleanly so Docker can restart the container.\n")
        sys.exit(1)
