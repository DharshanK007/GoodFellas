"""
skyguard/edge/esp32_scraper.py
──────────────────────────────────────────────────────────────────────────────
ESP32 Edge Station — Pull/Scrape Poller  (Phase 1 — isolated sanity-check only)
──────────────────────────────────────────────────────────────────────────────

Uses ONLY Python stdlib (urllib, asyncio, json) — zero new pip dependencies.

Isolation guarantees:
  • Does NOT import from skyguard.realtime.pipeline
  • Does NOT import from skyguard.data.ingestion_adapter
  • Does NOT touch DACM, Channel 1/2, or any pipeline model

Polling behaviour:
  • Poll interval: POLL_INTERVAL_S (default 7 s)
  • Timeout per request: REQUEST_TIMEOUT_S (default 4 s)
  • Ring buffer: last RING_BUFFER_SIZE readings (default 500)
  • On failure: scrape_status → "UNREACHABLE", last_error recorded.
    Old readings are NOT dropped — but the API clearly reports the outage.
"""

import os
import json
import asyncio
import logging
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger("skyguard.edge.scraper")

# ─── Configuration ────────────────────────────────────────────────────────────
EDGE_STATION_URL: str = os.environ.get(
    "EDGE_STATION_URL",
    "http://10.105.233.109/data",   # real ESP32 IP — update after checking Serial Monitor
)

POLL_INTERVAL_S:   float = float(os.environ.get("EDGE_POLL_INTERVAL_S",  "7"))
REQUEST_TIMEOUT_S: float = float(os.environ.get("EDGE_REQUEST_TIMEOUT_S", "4"))
RING_BUFFER_SIZE:  int   = int(os.environ.get(
    "EDGE_RING_BUFFER_SIZE",   "500"))

# ─── In-memory state ──────────────────────────────────────────────────────────
_ring_buffer: Deque[Dict[str, Any]] = deque(maxlen=RING_BUFFER_SIZE)

_scrape_status:          str           = "WAITING"   # WAITING | OK | UNREACHABLE
_last_success_at:        Optional[str] = None
_last_error:             Optional[str] = None
_total_scrape_attempts:  int           = 0
_total_scrape_successes: int           = 0

_poller_task: Optional[asyncio.Task] = None

# ─── Expected fields (non-fatal validation) ───────────────────────────────────
_EXPECTED_FIELDS = {
    "station_id", "timestamp_ms", "temperature_c", "humidity_pct",
    "pressure_hpa", "rain_detected", "rain_digital_raw", "rain_raw_analog",
    "rain_pct", "status", "composite_physics_inconsistency", "flagged",
    "dalton_violation", "r_virtual_temp_k", "r_vapor_pressure_hpa",
    "r_refractive_index",
}


# ─── Core scrape (runs in a thread-pool so it never blocks the event loop) ────
def _blocking_scrape() -> Dict[str, Any]:
    """
    Synchronous HTTP GET using stdlib urllib.
    Returns a dict with one of two shapes:
        {"ok": True,  "payload": {...}}
        {"ok": False, "error":   "description"}
    """
    req = urllib.request.Request(
        EDGE_STATION_URL,
        headers={
            "User-Agent": "SkyGuard-EdgeScraper/1.0",
            "Accept":     "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"URLError: {exc.reason}"}
    except OSError as exc:
        return {"ok": False, "error": f"OSError: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSON parse error: {exc}"}

    return {"ok": True, "payload": payload}


async def _scrape_once() -> None:
    """Run the blocking scrape in a thread, then update module-level state."""
    global _scrape_status, _last_success_at, _last_error
    global _total_scrape_attempts, _total_scrape_successes

    _total_scrape_attempts += 1
    received_at = datetime.now(timezone.utc).isoformat()

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _blocking_scrape)

    if not result["ok"]:
        _scrape_status = "UNREACHABLE"
        _last_error    = result["error"]
        logger.warning("[ESP32 scraper] %s", _last_error)
        return

    payload: Dict[str, Any] = result["payload"]

    missing = _EXPECTED_FIELDS - set(payload.keys())
    if missing:
        logger.warning("[ESP32 scraper] Partial payload — missing: %s", missing)

    _ring_buffer.append({**payload, "received_at": received_at})

    _scrape_status          = "OK"
    _last_success_at        = received_at
    _last_error             = None
    _total_scrape_successes += 1
    logger.info(
        "[ESP32 scraper] OK  temp=%.1f°C  hum=%.0f%%  pres=%.1fhPa  rain=%s  status=%s",
        payload.get("temperature_c", float("nan")),
        payload.get("humidity_pct",  float("nan")),
        payload.get("pressure_hpa",  float("nan")),
        payload.get("rain_pct",      "?"),
        payload.get("status",        "?"),
    )


# ─── Background poll loop ─────────────────────────────────────────────────────
async def _poll_loop() -> None:
    logger.info(
        "[ESP32 scraper] Starting — URL=%s  interval=%.0fs  buffer=%d",
        EDGE_STATION_URL, POLL_INTERVAL_S, RING_BUFFER_SIZE,
    )
    while True:
        await _scrape_once()
        await asyncio.sleep(POLL_INTERVAL_S)


# ─── Public control API ───────────────────────────────────────────────────────
def start_poller() -> None:
    """
    Launch the background polling task.
    Called from the FastAPI startup event.
    Safe to call multiple times — only one loop runs.
    """
    global _poller_task
    if _poller_task is not None and not _poller_task.done():
        return
    loop = asyncio.get_event_loop()
    _poller_task = loop.create_task(_poll_loop())
    logger.info("[ESP32 scraper] Background poller started.")


def stop_poller() -> None:
    global _poller_task
    if _poller_task and not _poller_task.done():
        _poller_task.cancel()
        logger.info("[ESP32 scraper] Background poller stopped.")


# ─── Query helpers (called by FastAPI route handlers) ────────────────────────
def get_latest_reading() -> Optional[Dict[str, Any]]:
    return _ring_buffer[-1] if _ring_buffer else None


def get_history(limit: int = 100) -> list:
    limit = max(1, min(limit, RING_BUFFER_SIZE))
    items = list(_ring_buffer)
    return list(reversed(items[-limit:]))


def get_scrape_state() -> Dict[str, Any]:
    return {
        "scrape_status":     _scrape_status,
        "last_success_at":   _last_success_at,
        "last_error":        _last_error,
        "total_attempts":    _total_scrape_attempts,
        "total_successes":   _total_scrape_successes,
        "buffered_readings": len(_ring_buffer),
        "poll_interval_s":   POLL_INTERVAL_S,
        "configured_url":    EDGE_STATION_URL,
    }
