"""
Open-Meteo ERA5 Wind Data Cache -- SkyGuard AI DACM Wind Fallback.

Primary wind source:  Station AWS observation (wind_speed / wind_direction fields).
Fallback wind source: Open-Meteo ERA5 historical hourly archive.

API: https://archive-api.open-meteo.com/v1/archive
Variables: wind_speed_10m (m/s), wind_direction_10m (degrees, meteorological FROM convention)

Provenance tags:
  STATION_OBSERVATION  -- direct from AWS obs
  OPEN_METEO_ERA5      -- fetched from archive-api.open-meteo.com (ERA5 reanalysis)
  UNAVAILABLE          -- neither source had valid wind data
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# In-memory cache: (lat_4dp, lon_4dp, hour_utc_iso) -> WindRecord dict
_WIND_CACHE: Dict[Tuple[str, str, str], Dict] = {}


def _round_key(lat: float, lon: float, dt: datetime) -> Tuple[str, str, str]:
    lat_r = round(lat, 4)
    lon_r = round(lon, 4)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_hour = dt.replace(minute=0, second=0, microsecond=0)
    return (str(lat_r), str(lon_r), dt_hour.isoformat())


def prefetch_wind_for_location(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> int:
    try:
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date   = end_dt.strftime("%Y-%m-%d")
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date={start_date}&end_date={end_date}"
            "&hourly=wind_speed_10m,wind_direction_10m"
            "&wind_speed_unit=ms&timezone=UTC"
        )
        logger.info("[WindCache] Prefetching ERA5 wind | lat=%.4f lon=%.4f | %s to %s", lat, lon, start_date, end_date)
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        hourly     = data.get("hourly", {})
        times      = hourly.get("time", [])
        speeds     = hourly.get("wind_speed_10m", [])
        directions = hourly.get("wind_direction_10m", [])
        count = 0
        for i, t_str in enumerate(times):
            try:
                dt_utc = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
                spd    = speeds[i]     if i < len(speeds)     else None
                dirn   = directions[i] if i < len(directions) else None
                if spd is None or dirn is None:
                    continue
                key = _round_key(lat, lon, dt_utc)
                _WIND_CACHE[key] = {
                    "wind_speed_ms": float(spd),
                    "wind_from_deg": float(dirn),
                    "wind_source":   "OPEN_METEO_ERA5",
                    "wind_timestamp": dt_utc.isoformat(),
                    "wind_match_method": "open_meteo_era5_nearest_hour",
                }
                count += 1
            except Exception:
                continue
        logger.info("[WindCache] Cached %d hourly ERA5 records lat=%.4f lon=%.4f", count, lat, lon)
        return count
    except Exception as exc:
        logger.warning("[WindCache] Open-Meteo prefetch failed (lat=%.4f lon=%.4f): %s", lat, lon, exc)
        return 0


def get_wind_for_location(lat: float, lon: float, dt: datetime) -> Optional[Dict]:
    return _WIND_CACHE.get(_round_key(lat, lon, dt))


def resolve_observation_wind(
    obs_lat: float, obs_lon: float,
    obs_wind_speed: Optional[float], obs_wind_direction: Optional[float],
    obs_timestamp: datetime,
) -> Tuple[Optional[float], Optional[float], str, str, str]:
    """
    Two-tier wind resolution.
    Returns: (wind_speed_ms, wind_from_deg, wind_source, wind_timestamp, wind_match_method)
    """
    # Tier 1: station observation
    if obs_wind_speed is not None and obs_wind_speed > 0.05 and obs_wind_direction is not None:
        ts_iso = (obs_timestamp.replace(tzinfo=timezone.utc) if obs_timestamp.tzinfo is None else obs_timestamp).isoformat()
        return float(obs_wind_speed), float(obs_wind_direction), "STATION_OBSERVATION", ts_iso, "station_observation_exact"
    # Tier 2: Open-Meteo ERA5 cache
    cached = get_wind_for_location(obs_lat, obs_lon, obs_timestamp)
    if cached:
        return cached["wind_speed_ms"], cached["wind_from_deg"], cached["wind_source"], cached["wind_timestamp"], cached["wind_match_method"]
    # Fallback
    return None, None, "UNAVAILABLE", "", "no_wind_data"


def cache_size() -> int:
    return len(_WIND_CACHE)


def clear_cache() -> None:
    _WIND_CACHE.clear()
