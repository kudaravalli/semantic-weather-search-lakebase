"""
National Weather Service HTTP client.

Provides:
- Reusable HTTP connections
- Automatic retries for transient failures
- NWS point/grid resolution
- NWS forecast retrieval
- NWS active alert retrieval
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger("weather-client")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NWS_BASE_URL = "https://api.weather.gov"

NWS_TIMEOUT = float(
    os.getenv("NWS_TIMEOUT", "15")
)

NWS_USER_AGENT = os.getenv(
    "NWS_USER_AGENT",
    "databricks-lakebase-weather-app",
)

NWS_RETRY_TOTAL = 3


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WeatherClient:
    """
    Reusable client for the National Weather Service API.

    A requests.Session is maintained for the lifetime of the client so
    HTTP connections can be reused across requests during one sync.
    """

    def __init__(
        self,
        timeout: float = NWS_TIMEOUT,
    ) -> None:
        self.timeout = timeout
        self.session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a configured HTTP session."""
        session = requests.Session()

        retry = Retry(
            total=NWS_RETRY_TOTAL,
            connect=NWS_RETRY_TOTAL,
            read=NWS_RETRY_TOTAL,
            status=NWS_RETRY_TOTAL,
            backoff_factor=0.5,
            status_forcelist=(
                429,
                500,
                502,
                503,
                504,
            ),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=20,
        )

        session.mount(
            "https://",
            adapter,
        )

        session.mount(
            "http://",
            adapter,
        )

        session.headers.update(
            {
                "User-Agent": NWS_USER_AGENT,
                "Accept": (
                    "application/geo+json, "
                    "application/json"
                ),
            }
        )

        return session

    def _get(
        self,
        path: str,
    ) -> dict[str, Any]:
        """
        Perform a GET request against api.weather.gov.

        Raises:
            requests.HTTPError: If NWS returns an HTTP error.
            requests.RequestException: For network failures.
            ValueError: If the response is not a JSON object.
        """
        url = f"{NWS_BASE_URL}{path}"

        logger.debug(
            "NWS GET %s",
            path,
        )

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected NWS response from {path}"
            )

        return payload

    # -----------------------------------------------------------------------
    # Grid resolution
    # -----------------------------------------------------------------------

    def resolve_grid_point(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """
        Resolve latitude/longitude to an NWS forecast grid.

        Returns:
            {
                "office": "LOT",
                "grid_x": 76,
                "grid_y": 73,
                "payload": {...}
            }
        """
        latitude = float(latitude)
        longitude = float(longitude)

        # NWS coordinates do not need excessive precision here.
        path = (
            f"/points/"
            f"{latitude:.4f},"
            f"{longitude:.4f}"
        )

        payload = self._get(path)

        properties = payload.get(
            "properties",
            {},
        )

        if not isinstance(properties, dict):
            raise ValueError(
                "Invalid NWS points response"
            )

        office = properties.get("gridId")
        grid_x = properties.get("gridX")
        grid_y = properties.get("gridY")

        if not office:
            raise ValueError(
                "NWS points response is missing gridId"
            )

        if grid_x is None or grid_y is None:
            raise ValueError(
                "NWS points response is missing grid coordinates"
            )

        return {
            "office": str(office),
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "payload": payload,
        }

    # -----------------------------------------------------------------------
    # Forecast
    # -----------------------------------------------------------------------

    def get_forecast(
        self,
        office: str,
        grid_x: int,
        grid_y: int,
    ) -> dict[str, Any]:
        """Retrieve an NWS forecast for a grid point."""
        office = str(office).strip()

        if not office:
            raise ValueError(
                "NWS office is required"
            )

        return self._get(
            f"/gridpoints/"
            f"{office}/"
            f"{int(grid_x)},"
            f"{int(grid_y)}/"
            "forecast"
        )

    # -----------------------------------------------------------------------
    # Active alerts
    # -----------------------------------------------------------------------

    def get_active_alerts(
        self,
    ) -> list[dict[str, Any]]:
        """
        Retrieve active NWS alerts.

        Returns:
            A list of GeoJSON alert feature dictionaries.
        """
        payload = self._get(
            "/alerts/active"
        )

        features = payload.get(
            "features",
            [],
        )

        if not isinstance(features, list):
            logger.warning(
                "Unexpected NWS active-alert response"
            )
            return []

        return [
            feature
            for feature in features
            if isinstance(feature, dict)
        ]

