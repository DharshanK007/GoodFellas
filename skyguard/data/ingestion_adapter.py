"""
SkyGuard AI — Pluggable Ingestion Adapter Layer.

Defines the AbstractIngestionAdapter interface so that any data source
(CSV file drop, JSON API poll, MQTT stream, Open-Meteo ERA5 archive, or
a future direct IMD telemetry socket) can be connected to the
SkyGuardPipeline without touching any ML or physics code.

Architecture:
    AWS SOURCE ──► ConcreteAdapter.stream() ──► Iterator[List[AWSObservation]]
                                                       │
                                                SkyGuardPipeline.process_observation()

Provenance contract (§12, §13):
    Every observation emitted by an adapter MUST carry:
      obs.source_metadata["data_mode"]   : "REAL_HISTORICAL" | "LIVE" | "SYNTHETIC" | "REPLAY"
      obs.source_metadata["adapter"]     : adapter class name
      obs.source_metadata["ingested_at"] : ISO UTC timestamp of when the record entered SkyGuard
"""

from __future__ import annotations

import csv
import json
import urllib.request
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from skyguard.data.schema import AWSObservation, CanonicalDataset, QualityFlag
from skyguard.data.loader import AWSDataLoader
from skyguard.realtime.pipeline import log_pipeline_event


# ── Base Contract ─────────────────────────────────────────────────────────────

class AbstractIngestionAdapter(ABC):
    """
    Abstract base class for all SkyGuard data source adapters.

    Subclass this and implement stream() to plug any live or historical
    AWS feed into the pipeline without altering any ML/physics code.
    """

    #: Human-readable source identifier (set in each subclass)
    SOURCE_NAME: str = "UNKNOWN"
    #: One of: REAL_HISTORICAL | LIVE | SYNTHETIC | REPLAY
    DATA_MODE: str = "UNKNOWN"

    def _tag(self, obs: AWSObservation) -> AWSObservation:
        """Stamps provenance metadata onto every emitted observation."""
        obs.source_metadata.setdefault("data_mode", self.DATA_MODE)
        obs.source_metadata.setdefault("adapter", type(self).__name__)
        obs.source_metadata.setdefault("source", self.SOURCE_NAME)
        obs.source_metadata["ingested_at"] = datetime.utcnow().isoformat() + "Z"
        return obs

    @abstractmethod
    def stream(self) -> Iterator[List[AWSObservation]]:
        """
        Yields batches of AWSObservation objects in chronological order.

        Each yielded list represents one discrete "time-slice" of the
        network (all stations at the same timestamp), which maps directly
        to SkyGuardPipeline.process_observation(obs, current_network_snapshot).

        Must:
          - Never yield an empty list.
          - Never silently swallow sensor gaps: flag missing fields with
            QualityFlag.MISSING_VALUE on the observation instead.
          - Always call self._tag(obs) before yielding.
        """
        raise NotImplementedError

    def as_canonical_dataset(self) -> CanonicalDataset:
        """
        Convenience: drains the adapter into a CanonicalDataset.
        Suitable for historical replay and training scenarios.
        """
        dataset = CanonicalDataset()
        for batch in self.stream():
            for obs in batch:
                dataset.add_observation(obs)
        return dataset


# ── Concrete Adapters ─────────────────────────────────────────────────────────

class CSVFileAdapter(AbstractIngestionAdapter):
    """
    Ingests a CSV file drop (e.g., hourly IMD AWS export) via AWSDataLoader.
    Columns are auto-detected from IMD, NOAA, or WMO synonym tables.

    Usage:
        adapter = CSVFileAdapter("/path/to/aws_feed.csv")
        for batch in adapter.stream():
            pipeline.process_observation(batch[0], batch)
    """

    SOURCE_NAME = "CSV_FILE"
    DATA_MODE = "REAL_HISTORICAL"

    def __init__(self, file_path: str, station_id_override: Optional[str] = None):
        self.file_path = Path(file_path)
        self.station_id_override = station_id_override

    def stream(self) -> Iterator[List[AWSObservation]]:
        log_pipeline_event(
            "INGESTION",
            f"[CSVFileAdapter] Loading: {self.file_path.name} | mode={self.DATA_MODE}"
        )
        dataset = AWSDataLoader.from_csv(self.file_path)
        if self.station_id_override:
            for obs in dataset.observations:
                obs.station_id = self.station_id_override

        # Group by timestamp and emit time-slices in order
        ts_map: dict = {}
        for obs in dataset.observations:
            self._tag(obs)
            ts_map.setdefault(obs.timestamp, []).append(obs)

        for ts in sorted(ts_map):
            yield ts_map[ts]


class JSONFileAdapter(AbstractIngestionAdapter):
    """
    Ingests a JSON file (array of observation dicts, or {"observations": [...]}).

    Supports the schema produced by Open-Meteo, IMD BUFR → JSON converters,
    or SkyGuard's own to_dict() export format.
    """

    SOURCE_NAME = "JSON_FILE"
    DATA_MODE = "REAL_HISTORICAL"

    def __init__(self, file_path: str, station_id_override: Optional[str] = None):
        self.file_path = Path(file_path)
        self.station_id_override = station_id_override

    def stream(self) -> Iterator[List[AWSObservation]]:
        log_pipeline_event(
            "INGESTION",
            f"[JSONFileAdapter] Loading: {self.file_path.name} | mode={self.DATA_MODE}"
        )
        dataset = AWSDataLoader.from_json(self.file_path)
        if self.station_id_override:
            for obs in dataset.observations:
                obs.station_id = self.station_id_override

        ts_map: dict = {}
        for obs in dataset.observations:
            self._tag(obs)
            ts_map.setdefault(obs.timestamp, []).append(obs)

        for ts in sorted(ts_map):
            yield ts_map[ts]


class OpenMeteoLiveAdapter(AbstractIngestionAdapter):
    """
    Fetches the most recent N hours of ERA5 reanalysis data from the
    Open-Meteo Historical Archive API for a given station coordinate.

    Primary use: plugging a real IMD station into the pipeline when no
    direct telemetry socket is available. Updates on each call to stream().

    Usage:
        adapter = OpenMeteoLiveAdapter(
            station_id="IMD-0342",
            lat=25.3478, lon=74.6348,
            hours=48,
        )
        dataset = adapter.as_canonical_dataset()
    """

    SOURCE_NAME = "OPEN_METEO_ERA5_LIVE"
    DATA_MODE = "REAL_HISTORICAL"

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        station_id: str,
        lat: float,
        lon: float,
        hours: int = 48,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.station_id = station_id
        self.lat = lat
        self.lon = lon
        self.hours = hours
        self.start_date = start_date
        self.end_date = end_date

    def _build_url(self) -> str:
        from datetime import timedelta
        if self.start_date and self.end_date:
            sd, ed = self.start_date, self.end_date
        else:
            now = datetime.utcnow()
            ed = now.strftime("%Y-%m-%d")
            sd = (now - timedelta(hours=self.hours)).strftime("%Y-%m-%d")

        params = {
            "latitude": f"{self.lat:.4f}",
            "longitude": f"{self.lon:.4f}",
            "start_date": sd,
            "end_date": ed,
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "surface_pressure,wind_speed_10m,wind_direction_10m"
            ),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        return self.BASE_URL + "?" + urllib.parse.urlencode(params)

    def stream(self) -> Iterator[List[AWSObservation]]:
        url = self._build_url()
        log_pipeline_event(
            "INGESTION",
            f"[OpenMeteoLiveAdapter] Fetching ERA5 for {self.station_id} "
            f"({self.lat:.4f}N, {self.lon:.4f}E) | {self.hours}h window"
        )

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "SkyGuard-AI-LiveAdapter/2.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            log_pipeline_event(
                "INGESTION",
                f"[OpenMeteoLiveAdapter] API fetch failed for {self.station_id}: {exc}",
                level="WARNING",
            )
            return

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rhs = hourly.get("relative_humidity_2m", [])
        press = hourly.get("surface_pressure", [])
        ws = hourly.get("wind_speed_10m", [])
        wd = hourly.get("wind_direction_10m", [])

        ts_map: dict = {}
        for i, t_str in enumerate(times):
            try:
                ts = datetime.fromisoformat(t_str)
                if ts.tzinfo:
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

                t_val = float(temps[i]) if i < len(temps) and temps[i] is not None else None
                p_val = float(press[i]) if i < len(press) and press[i] is not None else None
                rh_val = float(rhs[i]) if i < len(rhs) and rhs[i] is not None else None
                ws_val = float(ws[i]) if i < len(ws) and ws[i] is not None else None
                wd_val = float(wd[i]) if i < len(wd) and wd[i] is not None else None

                flags = []
                if t_val is None:
                    flags.append(QualityFlag.MISSING_VALUE)
                    t_val = 25.0
                if p_val is None:
                    flags.append(QualityFlag.MISSING_VALUE)
                    p_val = 1013.25
                if rh_val is None:
                    flags.append(QualityFlag.MISSING_VALUE)
                    rh_val = 50.0

                obs = AWSObservation(
                    station_id=self.station_id,
                    timestamp=ts,
                    latitude=self.lat,
                    longitude=self.lon,
                    temperature=t_val,
                    pressure=p_val,
                    humidity=rh_val,
                    wind_speed=ws_val,
                    wind_direction=wd_val,
                    quality_flags=flags,
                )
                self._tag(obs)
                ts_map.setdefault(ts, []).append(obs)
            except Exception:
                continue

        log_pipeline_event(
            "INGESTION",
            f"[OpenMeteoLiveAdapter] Ingested {len(ts_map)} hourly time-slices for {self.station_id}"
        )
        for ts in sorted(ts_map):
            yield ts_map[ts]


class MQTTStubAdapter(AbstractIngestionAdapter):
    """
    Stub adapter for a future MQTT / WebSocket live telemetry feed.

    Replace the body of stream() with actual paho-mqtt or websockets
    subscription logic when connecting to a real AWS MQTT broker.

    The stub currently reads from a pre-loaded list of AWSObservation
    objects, making it usable for integration testing without an actual
    broker.

    Usage (real broker, future):
        class MyMQTTAdapter(MQTTStubAdapter):
            def stream(self):
                import paho.mqtt.client as mqtt
                ...yield live batches...
    """

    SOURCE_NAME = "MQTT_LIVE_STUB"
    DATA_MODE = "LIVE"

    def __init__(self, preloaded_observations: Optional[list] = None):
        self._observations = preloaded_observations or []

    def stream(self) -> Iterator[List[AWSObservation]]:
        log_pipeline_event(
            "INGESTION",
            f"[MQTTStubAdapter] Streaming {len(self._observations)} pre-loaded "
            f"observations (stub — replace with real broker subscription)"
        )
        ts_map: dict = {}
        for obs in self._observations:
            self._tag(obs)
            ts_map.setdefault(obs.timestamp, []).append(obs)
        for ts in sorted(ts_map):
            yield ts_map[ts]


# ── Factory Helper ────────────────────────────────────────────────────────────

def make_adapter(source: str, **kwargs) -> AbstractIngestionAdapter:
    """
    Factory function for runtime adapter construction.

    Args:
        source: One of "csv" | "json" | "open_meteo" | "mqtt_stub"
        **kwargs: Forwarded to the chosen adapter constructor.

    Returns:
        A ready-to-use AbstractIngestionAdapter instance.

    Example:
        adapter = make_adapter("open_meteo", station_id="IMD-0342",
                               lat=25.35, lon=74.63, hours=72)
        for batch in adapter.stream():
            pipeline.process_observation(batch[0], batch)
    """
    _MAP = {
        "csv":        CSVFileAdapter,
        "json":       JSONFileAdapter,
        "open_meteo": OpenMeteoLiveAdapter,
        "mqtt_stub":  MQTTStubAdapter,
    }
    key = source.lower().strip()
    if key not in _MAP:
        raise ValueError(
            f"Unknown adapter source '{source}'. "
            f"Valid options: {list(_MAP.keys())}"
        )
    return _MAP[key](**kwargs)
