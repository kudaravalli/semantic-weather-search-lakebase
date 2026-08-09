"""
Weather synchronization pipeline.

Responsibilities:
- Resolve locations into coordinates.
- Cache geocoding and NWS grid information.
- Fetch NWS alerts and forecasts.
- Normalize API responses into weather documents.
- Generate semantic embeddings.
- Upsert documents and embeddings into Lakebase.

This module is independent of Flask and can be used by:
- app.py
- Databricks notebooks
- scheduled jobs
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import lakebase
import requests

from weather_client import WeatherClient
from weather_schema import (
    LOCATION_CACHE_TABLE,
    WEATHER_DOCUMENTS_TABLE,
    WEATHER_EMBEDDINGS_TABLE,
)
from embedding.chunking import chunk_text
from weather_search import (
    MODEL_NAME,
    get_embedding_model,
)

logger = logging.getLogger("weather-sync")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEOCODING_URL = (
    "https://nominatim.openstreetmap.org/search"
)

GEOCODING_TIMEOUT = 15

NOMINATIM_USER_AGENT = (
    "databricks-lakebase-weather-app"
)

MAX_SYNC_LIMIT = 500


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_weather_documents(
    locations: list[Any],
    limit: int = 50,
) -> int:
    """
    Synchronize weather data for the requested locations.

    Active NWS alerts are fetched once per synchronization run.
    Forecasts are fetched for each requested location.

    Returns:
        Number of documents successfully synchronized.
    """
    if not locations:
        return 0

    limit = max(
        1,
        min(int(limit), MAX_SYNC_LIMIT),
    )

    client = WeatherClient()

    documents: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Active alerts
    # -----------------------------------------------------------------------

    try:
        alerts = client.get_active_alerts()

        for alert in alerts[:limit]:
            documents.append(
                normalize_alert(
                    location=None,
                    alert=alert,
                )
            )

    except Exception:
        logger.exception(
            "Failed to synchronize NWS alerts"
        )

    # -----------------------------------------------------------------------
    # Location forecasts
    # -----------------------------------------------------------------------

    for location in locations:
        try:
            resolved = resolve_location(
                location
            )

            grid = resolve_nws_grid(
                client,
                resolved,
            )

            # Cache the resolved grid for string locations.
            #
            # Coordinate dictionaries are already explicit coordinates,
            # so there is no user-facing location key to cache.
            if isinstance(location, str):
                save_location_cache(
                    location.strip(),
                    {
                        **resolved,
                        "nws_office": grid["office"],
                        "nws_grid_x": grid["grid_x"],
                        "nws_grid_y": grid["grid_y"],
                        "payload": grid.get("payload"),
                    },
                )

            forecast = client.get_forecast(
                office=grid["office"],
                grid_x=grid["grid_x"],
                grid_y=grid["grid_y"],
            )

            periods = (
                forecast
                .get("properties", {})
                .get("periods", [])
            )

            if not isinstance(periods, list):
                logger.warning(
                    "Unexpected forecast response for %s",
                    location,
                )
                continue

            for period in periods[:limit]:
                if not isinstance(period, dict):
                    continue

                documents.append(
                    normalize_forecast(
                        location=resolved,
                        forecast=period,
                    )
                )

        except Exception:
            logger.exception(
                "Failed syncing location %s",
                location,
            )

    if not documents:
        logger.warning("No weather documents to synchronize")
        return 0

    logger.info(f"Upserting {len(documents)} weather documents")
    try:
        result = upsert_weather_documents(documents)
        # Defensive check: ensure result is a valid integer
        if result is None:
            logger.error("upsert_weather_documents returned None!")
            return 0
        logger.info(f"Successfully synchronized {result} weather documents")
        return result
    except Exception as e:
        logger.exception(f"Failed to upsert weather documents: {e}")
        # Return 0 to indicate failure but maintain API contract
        return 0


# ---------------------------------------------------------------------------
# Location resolution
# ---------------------------------------------------------------------------

def resolve_location(
    location: Any,
) -> dict[str, Any]:
    """
    Resolve a location into normalized coordinates.

    Supported inputs:

        "Chicago, IL"

    or:

        {
            "lat": 41.8781,
            "lon": -87.6298
        }
    """
    if isinstance(location, dict):
        if (
            "lat" not in location
            or "lon" not in location
        ):
            raise ValueError(
                "Location dictionary requires "
                "'lat' and 'lon'"
            )

        try:
            latitude = float(
                location["lat"]
            )
            longitude = float(
                location["lon"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Location latitude and longitude "
                "must be numeric"
            ) from exc

        if not (
            -90 <= latitude <= 90
        ):
            raise ValueError(
                "Latitude must be between -90 and 90"
            )

        if not (
            -180 <= longitude <= 180
        ):
            raise ValueError(
                "Longitude must be between -180 and 180"
            )

        return {
            "display_name": (
                location.get("name")
                or f"{latitude},{longitude}"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "nws_office": location.get(
                "nws_office"
            ),
            "nws_grid_x": location.get(
                "nws_grid_x"
            ),
            "nws_grid_y": location.get(
                "nws_grid_y"
            ),
            "payload": location.get(
                "payload"
            ),
        }

    if isinstance(location, str):
        location = location.strip()

        if not location:
            raise ValueError(
                "Location cannot be empty"
            )

        cached = get_cached_location(
            location
        )

        if cached:
            return cached

        coordinates = geocode_location(
            location
        )

        save_location_cache(
            location,
            coordinates,
        )

        return coordinates

    raise ValueError(
        "Unsupported location type: "
        f"{type(location).__name__}"
    )


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_location(
    location: str,
) -> dict[str, Any]:
    """
    Resolve a human-readable location using
    OpenStreetMap Nominatim.
    """
    response = requests.get(
        GEOCODING_URL,
        params={
            "q": location,
            "format": "json",
            "limit": 1,
        },
        headers={
            "User-Agent": NOMINATIM_USER_AGENT,
        },
        timeout=GEOCODING_TIMEOUT,
    )

    response.raise_for_status()

    results = response.json()

    if not isinstance(results, list):
        raise ValueError(
            "Unexpected geocoding response"
        )

    if not results:
        raise ValueError(
            f"Unable to geocode location: {location}"
        )

    result = results[0]

    try:
        latitude = float(
            result["lat"]
        )
        longitude = float(
            result["lon"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid geocoding response for {location}"
        ) from exc

    return {
        "display_name": result.get(
            "display_name",
            location,
        ),
        "latitude": latitude,
        "longitude": longitude,
    }


# ---------------------------------------------------------------------------
# Location cache
# ---------------------------------------------------------------------------

def get_cached_location(
    location_key: str,
) -> dict[str, Any] | None:
    """Return cached coordinates and NWS grid information."""
    rows = lakebase.run_query(
        f"""
        SELECT
            location_key,
            display_name,
            latitude,
            longitude,
            nws_office,
            nws_grid_x,
            nws_grid_y,
            payload
        FROM {LOCATION_CACHE_TABLE}
        WHERE location_key = %s
        """,
        (location_key,),
    )

    if not rows:
        return None

    row = rows[0]

    return {
        "display_name": row["display_name"],
        "latitude": float(
            row["latitude"]
        ),
        "longitude": float(
            row["longitude"]
        ),
        "nws_office": row.get(
            "nws_office"
        ),
        "nws_grid_x": row.get(
            "nws_grid_x"
        ),
        "nws_grid_y": row.get(
            "nws_grid_y"
        ),
        "payload": row.get(
            "payload"
        ),
    }


def save_location_cache(
    location_key: str,
    location: dict[str, Any],
) -> None:
    """Insert or update a cached location."""
    lakebase.run_write(
        f"""
        INSERT INTO {LOCATION_CACHE_TABLE} (
            location_key,
            latitude,
            longitude,
            display_name,
            nws_office,
            nws_grid_x,
            nws_grid_y,
            payload,
            updated_at
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            now()
        )
        ON CONFLICT (location_key)
        DO UPDATE SET
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            display_name = EXCLUDED.display_name,
            nws_office = EXCLUDED.nws_office,
            nws_grid_x = EXCLUDED.nws_grid_x,
            nws_grid_y = EXCLUDED.nws_grid_y,
            payload = EXCLUDED.payload,
            updated_at = now()
        """,
        (
            location_key,
            location["latitude"],
            location["longitude"],
            location.get(
                "display_name"
            ),
            location.get(
                "nws_office"
            ),
            location.get(
                "nws_grid_x"
            ),
            location.get(
                "nws_grid_y"
            ),
            json.dumps(
                location.get("payload")
            )
            if location.get("payload")
            is not None
            else None,
        ),
    )


# ---------------------------------------------------------------------------
# NWS grid resolution
# ---------------------------------------------------------------------------

def resolve_nws_grid(
    client: WeatherClient,
    location: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve coordinates into an NWS forecast grid.

    If the location already contains cached grid information, avoid
    another /points request.
    """
    cached_office = location.get(
        "nws_office"
    )
    cached_x = location.get(
        "nws_grid_x"
    )
    cached_y = location.get(
        "nws_grid_y"
    )

    if (
        cached_office
        and cached_x is not None
        and cached_y is not None
    ):
        return {
            "office": str(cached_office),
            "grid_x": int(cached_x),
            "grid_y": int(cached_y),
            "payload": location.get(
                "payload"
            ),
        }

    return client.resolve_grid_point(
        location["latitude"],
        location["longitude"],
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_alert(
    location: dict[str, Any] | None,
    alert: dict[str, Any],
) -> dict[str, Any]:
    """Normalize an NWS alert into a weather document."""
    properties = alert.get(
        "properties",
        {},
    )

    if not isinstance(properties, dict):
        properties = {}

    area_description = properties.get(
        "areaDesc"
    )

    if area_description:
        location_value = str(
            area_description
        )
    elif location:
        location_value = location[
            "display_name"
        ]
    else:
        location_value = "NWS Active Alert"

    narrative = (
        properties.get("description")
        or properties.get("instruction")
        or ""
    )

    return build_document(
        location=location_value,
        source_type="alert",
        headline=properties.get(
            "event"
        ),
        narrative=str(narrative),
        issued_at=properties.get(
            "sent"
        ),
        effective_at=properties.get(
            "effective"
        ),
        payload=alert,
    )


def normalize_forecast(
    location: dict[str, Any],
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """Normalize an NWS forecast period."""
    return build_document(
        location=location[
            "display_name"
        ],
        source_type="forecast",
        headline=forecast.get(
            "name"
        ),
        narrative=forecast.get(
            "detailedForecast",
            "",
        ),
        issued_at=None,
        effective_at=None,
        payload=forecast,
    )


def build_document(
    location: str,
    source_type: str,
    headline: str | None,
    narrative: str,
    issued_at: str | None,
    effective_at: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a normalized, deterministic weather document.

    The document ID is based on the normalized payload. Identical
    payloads therefore produce identical IDs and are naturally
    idempotent during synchronization.
    """
    raw_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    content_hash = hashlib.sha256(
        raw_payload.encode("utf-8")
    ).hexdigest()

    document_key = "|".join(
        (
            location,
            source_type,
            content_hash,
        )
    )

    document_id = hashlib.sha256(
        document_key.encode("utf-8")
    ).hexdigest()

    return {
        "id": document_id,
        "location": location,
        "source_type": source_type,
        "headline": headline,
        "narrative_text": narrative or "",
        "content_hash": content_hash,
        "issued_at": issued_at,
        "effective_at": effective_at,
        "payload": payload,
        "synced_at": datetime.now(
            timezone.utc
        ),
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_embedding_text(
    document: dict[str, Any],
) -> str:
    """
    Create the text representation used for semantic embedding.
    """
    headline = (
        document.get("headline")
        or ""
    )

    narrative = (
        document.get("narrative_text")
        or ""
    )

    return "\n\n".join(
        part.strip()
        for part in (
            str(headline),
            str(narrative),
        )
        if part and str(part).strip()
    )


def build_embedding_id(
    document_id: str,
    chunk_text: str,
) -> str:
    """Create a deterministic ID for an embedding."""
    chunk_hash = hashlib.sha256(
        chunk_text.encode("utf-8")
    ).hexdigest()

    return hashlib.sha256(
        f"{document_id}|{chunk_hash}".encode(
            "utf-8"
        )
    ).hexdigest()


# ---------------------------------------------------------------------------
# Existing embedding lookup
# ---------------------------------------------------------------------------

def _get_existing_embedding_hashes(
    cursor: Any,
    document_ids: list[str],
) -> dict[str, str]:
    """
    Retrieve existing embedding content hashes.

    This prevents unnecessary SentenceTransformer work when the
    searchable document content has not changed.
    """
    if not document_ids:
        return {}

    placeholders = ",".join(
        ["%s"] * len(document_ids)
    )

    cursor.execute(
        f"""
        SELECT
            document_id,
            content_hash
        FROM {WEATHER_EMBEDDINGS_TABLE}
        WHERE document_id IN ({placeholders})
        """,
        tuple(document_ids),
    )

    rows = cursor.fetchall()

    return {
        row["document_id"]: row[
            "content_hash"
        ]
        for row in rows
        if row.get("content_hash")
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_weather_documents(
    documents: list[dict[str, Any]],
) -> int:
    """
    Upsert weather documents and semantic embeddings.

    Documents and embeddings are written within the same database
    transaction.

    Embeddings are generated in batches rather than one document at
    a time, which substantially reduces SentenceTransformer overhead.
    """
    if not documents:
        logger.info("No documents to upsert")
        return 0

    logger.info(f"Starting upsert of {len(documents)} documents")
    count = 0

    with lakebase.get_connection() as conn:
        try:
            with conn.cursor() as cur:
                # -----------------------------------------------------------
                # Existing embedding state
                # -----------------------------------------------------------

                document_ids = [
                    document["id"]
                    for document in documents
                ]

                existing_hashes = (
                    _get_existing_embedding_hashes(
                        cur,
                        document_ids,
                    )
                )

                # -----------------------------------------------------------
                # Prepare documents and embedding work
                # -----------------------------------------------------------

                embedding_documents: list[
                    tuple[
                        dict[str, Any],
                        int,
                        str,
                    ]
                ] = []

                documents_processed = 0
                for document in documents:
                    documents_processed += 1
                    cur.execute(
                        f"""
                        INSERT INTO {WEATHER_DOCUMENTS_TABLE} (
                            id,
                            location,
                            source_type,
                            headline,
                            narrative_text,
                            content_hash,
                            issued_at,
                            effective_at,
                            payload,
                            synced_at
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            now()
                        )
                        ON CONFLICT (id)
                        DO UPDATE SET
                            location =
                                EXCLUDED.location,
                            source_type =
                                EXCLUDED.source_type,
                            headline =
                                EXCLUDED.headline,
                            narrative_text =
                                EXCLUDED.narrative_text,
                            content_hash =
                                EXCLUDED.content_hash,
                            issued_at =
                                EXCLUDED.issued_at,
                            effective_at =
                                EXCLUDED.effective_at,
                            payload =
                                EXCLUDED.payload,
                            synced_at =
                                now()
                        """,
                        (
                            document["id"],
                            document["location"],
                            document["source_type"],
                            document["headline"],
                            document[
                                "narrative_text"
                            ],
                            document[
                                "content_hash"
                            ],
                            document[
                                "issued_at"
                            ],
                            document[
                                "effective_at"
                            ],
                            json.dumps(
                                document["payload"]
                            ),
                        ),
                    )

                    embedding_text = (
                        build_embedding_text(
                            document
                        )
                    )

                    if not embedding_text:
                        logger.warning(
                            "Skipping empty embedding "
                            "for document %s",
                            document["id"],
                        )
                        continue

                    # The embedding table stores the document's
                    # content hash. If it is unchanged, there is
                    # no reason to invoke the ML model again.
                    if (
                        existing_hashes.get(
                            document["id"]
                        )
                        == document[
                            "content_hash"
                        ]
                    ):
                        count += 1
                        continue

                    # Chunk the document text using sliding window
                    chunks = chunk_text(
                        embedding_text
                    )

                    for chunk_index, chunk in enumerate(chunks):
                        embedding_documents.append(
                            (
                                document,
                                chunk_index,
                                chunk,
                            )
                        )

                # -----------------------------------------------------------
                # Batch embedding generation
                # -----------------------------------------------------------

                if embedding_documents:
                    model = get_embedding_model()

                    texts = [
                        chunk
                        for _, _, chunk
                        in embedding_documents
                    ]

                    logger.info(
                        f"Encoding {len(texts)} chunks "
                        f"from {len(set(doc['id'] for doc, _, _ in embedding_documents))} "
                        f"documents in batch"
                    )
                    vectors = model.encode(
                        texts,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        batch_size=64,  # Increased from 32 for better throughput
                        show_progress_bar=False,
                    )
                    logger.info(f"Encoding complete")

                    # -------------------------------------------------------
                    # Upsert embeddings
                    # -------------------------------------------------------

                    for (
                        index,
                        (
                            document,
                            chunk_index,
                            chunk,
                        ),
                    ) in enumerate(
                        embedding_documents
                    ):
                        vector = vectors[index]

                        embedding_id = (
                            build_embedding_id(
                                document["id"],
                                chunk,
                            )
                        )

                        vector_string = (
                            "["
                            + ",".join(
                                str(float(value))
                                for value in vector
                            )
                            + "]"
                        )

                        cur.execute(
                            f"""
                            INSERT INTO
                                {WEATHER_EMBEDDINGS_TABLE} (
                                    id,
                                    document_id,
                                    location,
                                    source_type,
                                    headline,
                                    chunk_index,
                                    chunk_text,
                                    content_hash,
                                    embedding,
                                    model_name,
                                    created_at
                                )
                            VALUES (
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s::vector,
                                %s,
                                now()
                            )
                            ON CONFLICT (document_id, chunk_index)
                            DO UPDATE SET
                                id =
                                    EXCLUDED.id,
                                document_id =
                                    EXCLUDED.document_id,
                                location =
                                    EXCLUDED.location,
                                source_type =
                                    EXCLUDED.source_type,
                                headline =
                                    EXCLUDED.headline,
                                chunk_index =
                                    EXCLUDED.chunk_index,
                                chunk_text =
                                    EXCLUDED.chunk_text,
                                content_hash =
                                    EXCLUDED.content_hash,
                                embedding =
                                    EXCLUDED.embedding,
                                model_name =
                                    EXCLUDED.model_name
                            """,
                            (
                                embedding_id,
                                document["id"],
                                document["location"],
                                document["source_type"],
                                document["headline"],
                                chunk_index,
                                chunk,
                                document[
                                    "content_hash"
                                ],
                                vector_string,
                                MODEL_NAME,
                            ),
                        )

                        count += 1

            conn.commit()
            logger.info(f"Transaction committed successfully, count={count}")

        except Exception as e:
            conn.rollback()

            logger.exception(
                f"Weather document transaction failed. "
                f"Processed {documents_processed}/{len(documents)} documents, "
                f"successfully upserted {count} documents before failure."
            )
            
            # Return partial count to provide feedback on what succeeded
            # before the failure, rather than raising which returns nothing
            logger.warning(
                f"Returning partial count ({count}/{documents_processed} docs) "
                f"due to transaction failure: {e}"
            )
            return count

    logger.info(f"Upsert complete, returning count={count}")
    return count

