"""
Data ingestion, canonical schemas, preprocessing, and anomaly injection.
"""

from skyguard.data.schema import (
    AWSObservation,
    StationMetadata,
    QualityFlag,
    CanonicalDataset,
    CorrectedObservation,
)
from skyguard.data.loader import AWSDataLoader
from skyguard.data.preprocessing import AWSPreprocessor
from skyguard.data.windowing import TemporalWindowGenerator
from skyguard.data.anomaly_injection import AnomalyInjector, AnomalyType

__all__ = [
    "AWSObservation",
    "StationMetadata",
    "QualityFlag",
    "CanonicalDataset",
    "CorrectedObservation",
    "AWSDataLoader",
    "AWSPreprocessor",
    "TemporalWindowGenerator",
    "AnomalyInjector",
    "AnomalyType",
]
