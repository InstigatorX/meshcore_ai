#!/usr/bin/env python3
"""
MeshCore -> LLM channel bot (Gemini OR local LLM) + TCP OR USB/Serial transport
+ separates AI channels and Weather channels
+ Scheduled daily weather broadcasts and on-demand `!weather [location]` fetching
+ Includes today's forecast (High/Low/Rain chance)
+ Configurable polling for US National Weather Service (NWS) severe weather alerts
+ Global `!ping` command with configurable template (returns SNR, RSSI, and Hops)
+ Global `!help` command listing available triggers based on context

DM replies require a destination with full 'public_key'. We resolve DM sender via
pubkey_prefix by caching contacts. DMs support all commands.

Also fixes a race where duplicate inbound packets could invoke the LLM twice by
locking the dedupe check.

Reconnect model:
- MeshCore internal auto_reconnect is DISABLED
- This script owns reconnect logic
- On disconnect / failed health checks, the whole session is rebuilt
"""

import asyncio
import os
import re
import time
import urllib.parse
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

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
    """Find multiple channel names. Returns {idx: normalized_name}."""
    if not channel_names:
        return {}

    want = {normalize_channel_name(c) for c in channel_names if c.strip()}
    if not want:
        return {}

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
        self.weather_units = weather_units.upper()
        self._seen_alerts: Set[str] = set()

        self.trigger_re = re.compile(rf"(^|\s+){re.escape(self.trigger)}(\s+|$)", re.IGNORECASE)

        self.history: Dict[int, Deque[Tuple[str, str]]] = {
            idx: deque(maxlen=history_turns * 2) for idx in ai_channels.keys()
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

            rain_chances = [int(h.get("chanceofrain", "0")) for h in today.get("hourly", [])]
            snow_chances = [int(h.get("chanceofsnow", "0")) for h in today.get("hourly", [])]
            max_rain = max(rain_chances) if rain_chances else 0
            max_snow = max(snow_chances) if snow_chances else 0

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

            if max_snow > 0:
                msg += f", Snow {max_snow}%"
            elif max_rain > 0:
                msg += f", Rain {max_rain}%"

            return msg

        except Exception as e:
            if self.debug:
                print(f"[DBG] Weather fetch error for {location}: {e}")
            return f"Could not find weather for '{location}' (API Error)."

    async def scheduled_weather_loop(self, stop_event: asyncio.Event) -> None:
        if not self.weather_times or not self.weather_location or not self.weather_channels:
            return

        parsed_times = []
        for t_str in self.weather_times:
            parts = t_str.split(":")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                parsed_times.append((int(parts[0]), int(parts[1])))

        if not parsed_times:
            return

        last_sent_min = -1

        while not stop_event.is_set():
            now = time.localtime()
            current_time = (now.tm_hour, now.tm_min)

            if current_time in parsed_times and now.tm_min != last_sent_min:
                last_sent_min = now.tm_min

                if self.debug:
                    print(f"[DBG] Triggering scheduled weather for {current_time}")

                ans = await self.fetch_weather(self.weather_location)

                async with self._llm_lock:
                    parts = chunk_text(ans, self.max_reply_chars)
                    for ch_idx in self.weather_channels.keys():
                        for i, part in enumerate(parts, start=1):
                            msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                            await self.mesh.commands.send_chan_msg(ch_idx, msg)
                            await asyncio.sleep(2)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass

    async def weather_alerts_loop(self, zones: str, interval_m: float, stop_event: asyncio.Event) -> None:
        if not zones or interval_m <= 0 or not self.weather_channels:
            return

        interval_s = interval_m * 60.0
        headers = {"User-Agent": "MeshCore-Weather-Alert-Bot/1.0 (https://github.com/)"}
        url = f"https://api.weather.gov/alerts/active?zone={urllib.parse.quote(zones)}"

        while not stop_event.is_set():
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

                    async with self._llm_lock:
                        parts = chunk_text(msg, self.max_reply_chars)
                        for ch_idx in self.weather_channels.keys():
                            for i, part in enumerate(parts, start=1):
                                out_msg = part if len(parts) == 1 else f"({i}/{len(parts)}) {part}"
                                await self.mesh.commands.send_chan_msg(ch_idx, out_msg)
                                await asyncio.sleep(2)

            except Exception as e:
                if self.debug:
                    print(f"[DBG] Weather alert polling error: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass

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
        snr = p.get("SNR") or p.get("rxSnr") or p.get("rx_snr") or p.get("snr")
        rssi = p.get("RSSI") or p.get("rxRssi") or p.get("rx_rssi") or p.get("rssi")
        path_len = p.get("path_len")

        hops = None
        if path_len is not None:
            hops = path_len
        else:
            hop_limit = p.get("hopLimit") or p.get("hop_limit")
            hop_start = p.get("hopStart") or p.get("hop_start")
            if hop_limit is not None:
                try:
                    hl = int(hop_limit)
                    hs = int(hop_start) if hop_start is not None else hl
                    hops = (hs - hl) if hs >= hl else 0
                except (ValueError, TypeError):
                    pass

        safe_snr = snr if snr is not None else "?"
        safe_rssi = rssi if rssi is not None else "?"
        safe_hops = hops if hops is not None else "?"

        try:
            return self.ping_template.format(snr=safe_snr, rssi=safe_rssi, hops=safe_hops)
        except Exception as e:
            if self.debug:
                print(f"[DBG] PING_TEMPLATE formatting error: {e}")
            return f"pong [SNR: {safe_snr}, RSSI: {safe_rssi}dBm, Hops: {safe_hops}]"

    # ---------------- Event handlers ----------------

    async def on_channel_msg(self, ev) -> None:
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

        h_trigger = self.help_trigger.lower()
        if body_lower == h_trigger or body_lower.startswith(h_trigger + " "):
            reply_text = self.get_help_string(is_ai_chan, is_weather_chan)
            out = self.format_chan_reply(sender, reply_text)
            if out:
                await self.mesh.commands.send_chan_msg(ch_idx, out)
            return

        p_trigger = self.ping_trigger.lower()
        if body_lower == p_trigger or body_lower.startswith(p_trigger + " "):
            reply_text = self.get_telemetry_string(p)
            out = self.format_chan_reply(sender, reply_text)
            if out:
                await self.mesh.commands.send_chan_msg(ch_idx, out)
            return

        if is_weather_chan:
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

        body_lower = body.lower()

        h_trigger = self.help_trigger.lower()
        if body_lower == h_trigger or body_lower.startswith(h_trigger + " "):
            reply_text = self.get_help_string(is_ai=True, is_weather=True)
            out = self.format_dm_reply(reply_text)
            if out:
                await self.mesh.commands.send_msg(dst, out)
            return

        p_trigger = self.ping_trigger.lower()
        if body_lower == p_trigger or body_lower.startswith(p_trigger + " "):
            reply_text = self.get_telemetry_string(p)
            out = self.format_dm_reply(reply_text)
            if out:
                await self.mesh.commands.send_msg(dst, out)
            return

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

        user = self.extract_after_trigger(body)
        if not user:
            return

        async with self._llm_lock:
            dummy_idx = 999
            if dummy_idx not in self.history:
                ml = self.history[list(self.ai_channels.keys())[0]].maxlen if self.ai_channels else 12
                self.history[dummy_idx] = deque(maxlen=ml)

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

        if hasattr(MeshCore, "create_serial"):
            mesh = await MeshCore.create_serial(serial_port, baud, auto_reconnect=False)  # type: ignore[attr-defined]
            if mesh is None:
                raise RuntimeError("MeshCore.create_serial() returned None")
            return mesh

        for alt in ("create_uart", "create_usb", "create_serial_port"):
            if hasattr(MeshCore, alt):
                fn = getattr(MeshCore, alt)
                try:
                    mesh = await fn(serial_port, baud, auto_reconnect=False)
                except TypeError:
                    mesh = await fn(serial_port, auto_reconnect=False)
                if mesh is None:
                    raise RuntimeError(f"MeshCore.{alt}() returned None")
                return mesh

        raise RuntimeError(
            "Your meshcore package does not expose MeshCore.create_serial (or known alternates). "
            "Run: python -c \"from meshcore import MeshCore; print([m for m in dir(MeshCore) if 'create' in m])\""
        )

    raise RuntimeError("MESHCORE_TRANSPORT must be one of: tcp | serial")


async def run_bot_once(llm: LLMClient) -> None:
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
    debug = env_str("DEBUG", "0").lower() in ("1", "true", "yes")
    system_prompt = env_str("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    weather_location = env_str("WEATHER_LOCATION", "")
    weather_times_raw = env_str("WEATHER_SCHEDULED_TIMES", "")
    weather_times = [t.strip() for t in weather_times_raw.split(",") if t.strip()]
    weather_trigger = env_str("WEATHER_TRIGGER", "!weather").strip()
    weather_units = env_str("WEATHER_UNITS", "F")

    alerts_zones = env_str("WEATHER_ALERTS_NWS_ZONES", "")
    alerts_interval_m = env_float("WEATHER_ALERTS_POLL_INTERVAL_M", 15.0)

    healthcheck_interval_s = env_int("HEALTHCHECK_INTERVAL_S", 10)
    healthcheck_max_failures = env_int("HEALTHCHECK_MAX_FAILURES", 3)

    print("[INFO] connecting to MeshCore...")
    mesh = await create_mesh_connection()
    await asyncio.sleep(1.0)

    if mesh is None:
        raise RuntimeError("MeshCore connection failed: got None")

    await mesh.start_auto_message_fetching()

    ai_channel_map = await resolve_channels(mesh, target_ai_channels, max_channels=scan_max)
    weather_channel_map = await resolve_channels(mesh, target_weather_channels, max_channels=scan_max)

    print("[OK] AI Channel map:")
    if not ai_channel_map:
        print("  (None configured)")
    for idx, name in ai_channel_map.items():
        print(f"  idx={idx} -> {name}")

    print("[OK] Weather Channel map:")
    if not weather_channel_map:
        print("  (None configured)")
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
        weather_units=weather_units
    )

    mesh.subscribe(EventType.CONTACTS, bot.on_contacts_event)
    mesh.subscribe(EventType.NEW_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.NEXT_CONTACT, bot.on_contacts_event)
    mesh.subscribe(EventType.CHANNEL_MSG_RECV, bot.on_channel_msg)
    mesh.subscribe(EventType.CONTACT_MSG_RECV, bot.on_dm_msg)

    await bot.refresh_contacts_best_effort()

    stop_event = asyncio.Event()
    background_tasks: List[asyncio.Task] = []

    if weather_times and weather_location and weather_channel_map:
        background_tasks.append(asyncio.create_task(bot.scheduled_weather_loop(stop_event)))
        print(f"[OK] Scheduled weather enabled: Location='{weather_location}', Times={weather_times}, Units={weather_units}")

    if alerts_zones and alerts_interval_m > 0 and weather_channel_map:
        background_tasks.append(asyncio.create_task(bot.weather_alerts_loop(alerts_zones, alerts_interval_m, stop_event)))
        print(f"[OK] NWS weather alerts polling enabled: Zones='{alerts_zones}', Interval={alerts_interval_m}m")

    print(f"\n[OK] Connected and Listening.")
    print(f"[TEST] Help      (Any channel or DM):  '{help_trigger}'")
    print(f"[TEST] Ping      (Any channel or DM):  '{ping_trigger}'")
    print(f"[TEST] AI Query  (AI Channels or DM):  '{trigger} hello'")
    print(f"[TEST] Weather   (Weather Ch. or DM):  '{weather_trigger}' or '{weather_trigger} Paris'\n")

    loop = asyncio.get_running_loop()
    disconnect_future: asyncio.Future[None] = loop.create_future()

    async def disconnected_handler(ev) -> None:
        print("[WARN] MeshCore disconnected event received")
        if not disconnect_future.done():
            disconnect_future.set_result(None)

    mesh.subscribe(EventType.DISCONNECTED, disconnected_handler)

    monitored_health_idx = None
    if ai_channel_map:
        monitored_health_idx = next(iter(ai_channel_map.keys()))
    elif weather_channel_map:
        monitored_health_idx = next(iter(weather_channel_map.keys()))

    async def health_monitor() -> None:
        if monitored_health_idx is None:
            return

        consecutive_failures = 0

        while not disconnect_future.done():
            try:
                ev = await mesh.commands.get_channel(monitored_health_idx)

                if ev is None or ev.type == EventType.ERROR:
                    consecutive_failures += 1
                    print(
                        f"[WARN] health check failed on channel idx {monitored_health_idx} "
                        f"({consecutive_failures}/{healthcheck_max_failures})"
                    )
                else:
                    if consecutive_failures and debug:
                        print("[DBG] health check recovered")
                    consecutive_failures = 0

                if consecutive_failures >= healthcheck_max_failures:
                    print(
                        f"[WARN] health check failed {consecutive_failures} times in a row; forcing reconnect"
                    )
                    if not disconnect_future.done():
                        disconnect_future.set_result(None)
                    return

            except Exception as e:
                consecutive_failures += 1
                print(
                    f"[WARN] health check exception ({consecutive_failures}/{healthcheck_max_failures}): {e}"
                )

                if consecutive_failures >= healthcheck_max_failures:
                    print("[WARN] too many consecutive health check exceptions; forcing reconnect")
                    if not disconnect_future.done():
                        disconnect_future.set_result(None)
                    return

            try:
                await asyncio.wait_for(disconnect_future, timeout=healthcheck_interval_s)
            except asyncio.TimeoutError:
                pass

    health_task = asyncio.create_task(health_monitor())

    try:
        await disconnect_future
    finally:
        stop_event.set()

        for task in background_tasks:
            task.cancel()
        health_task.cancel()

        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if debug:
                    print(f"[DBG] background task ended with error: {e}")

        try:
            await health_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if debug:
                print(f"[DBG] health task ended with error: {e}")

        print("[INFO] Closing MeshCore connection...")
        try:
            if hasattr(mesh, "aclose"):
                await mesh.aclose()
            elif hasattr(mesh, "close"):
                mesh.close()
        except Exception as e:
            if debug:
                print(f"[DBG] error while closing MeshCore connection: {e}")


async def main() -> None:
    backend = env_str("LLM_BACKEND", "gemini").lower()

    llm: LLMClient
    if backend == "gemini":
        api_key = env_str("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (required for LLM_BACKEND=gemini)")
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

    reconnect_delay = env_int("RECONNECT_DELAY_S", 5)
    reconnect_max_delay = env_int("RECONNECT_MAX_DELAY_S", 60)
    delay = reconnect_delay

    try:
        while True:
            try:
                await run_bot_once(llm)
                delay = reconnect_delay
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[WARN] bot session ended unexpectedly: {e}")

            print(f"[INFO] reconnecting in {delay}s...")
            await asyncio.sleep(delay)
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
