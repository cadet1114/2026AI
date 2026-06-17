from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any

import requests


USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def map_usgs_hazard_intensity(magnitude: float, depth_km: float) -> str:
    """Map public earthquake attributes to a coarse local hazard anchor."""
    effective = float(magnitude)
    if depth_km >= 200:
        effective -= 0.8
    elif depth_km >= 70:
        effective -= 0.35
    if effective < 5.0:
        return "low"
    if effective < 6.5:
        return "medium"
    return "high"


def fetch_usgs_catalog(
    output_path: str | Path,
    *,
    start_time: str = "2024-01-01",
    end_time: str = "2025-12-31",
    minimum_magnitude: float = 4.5,
    limit: int = 2000,
    timeout: int = 60,
) -> dict[str, Any]:
    """Download a bounded official USGS CSV and return reproducibility metadata."""
    params = {
        "format": "csv",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": minimum_magnitude,
        "orderby": "time",
        "limit": limit,
    }
    response = requests.get(USGS_QUERY_URL, params=params, timeout=timeout)
    response.raise_for_status()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    rows = list(csv.DictReader(io.StringIO(response.text)))
    return {
        "source": "USGS Earthquake Catalog API",
        "url": response.url,
        "license_note": "USGS-authored data are generally public domain; retain source attribution.",
        "downloaded_rows": len(rows),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "query": params,
    }


def load_usgs_catalog(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not raw.get("mag") or not raw.get("depth"):
                continue
            rows.append(
                {
                    "id": raw.get("id", ""),
                    "time": raw.get("time", ""),
                    "place": raw.get("place", ""),
                    "mag": float(raw["mag"]),
                    "depth": float(raw["depth"]),
                    "latitude": float(raw["latitude"]),
                    "longitude": float(raw["longitude"]),
                }
            )
    return rows

