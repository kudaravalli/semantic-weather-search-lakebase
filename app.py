"""
Lakebase Weather RAG Assistant Service App.

Provides:
- Flask REST API
- Lakebase persistence
- National Weather Service ingestion
- Weather document synchronization
- Semantic weather search using pgvector embeddings

Run locally:
    python app.py

Production:
    Run with Gunicorn through Databricks Apps.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import lakebase
from flask import Flask, jsonify, render_template, request

from weather_schema import (
    WEATHER_DOCUMENTS_TABLE,
    ensure_weather_schema,
)
from weather_search import (
    embed_query,
    load_embedding_model,
    search_weather_documents,
)
from weather_sync import sync_weather_documents


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("weather-app")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_WEATHER_LOCATIONS: list[Any] = [
    "Chicago, IL",
    "Austin, TX",
    {
        "lat": 37.7749,
        "lon": -122.4194,
    },
]

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

DEFAULT_SYNC_LIMIT = 50
MAX_SYNC_LIMIT = 500

DEFAULT_DOCUMENT_LIMIT = 100
MAX_DOCUMENT_LIMIT = 500

VALID_SOURCE_TYPES = frozenset(
    {
        "alert",
        "forecast",
        "observation",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_default_locations() -> list[Any]:
    """Load configured weather locations or return defaults."""
    configured = os.getenv("WEATHER_LOCATIONS_JSON")

    if not configured:
        return DEFAULT_WEATHER_LOCATIONS

    try:
        locations = json.loads(configured)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid WEATHER_LOCATIONS_JSON; "
            "using default locations"
        )
        return DEFAULT_WEATHER_LOCATIONS

    if not isinstance(locations, list):
        logger.warning(
            "WEATHER_LOCATIONS_JSON must contain a JSON list; "
            "using default locations"
        )
        return DEFAULT_WEATHER_LOCATIONS

    if not locations:
        logger.warning(
            "WEATHER_LOCATIONS_JSON is empty; "
            "using default locations"
        )
        return DEFAULT_WEATHER_LOCATIONS

    return locations


def _parse_positive_int(
    value: Any,
    default: int,
    maximum: int,
) -> int:
    """Parse and clamp a positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    return max(
        1,
        min(parsed, maximum),
    )


def _validate_source_type(
    source_type: Any,
) -> str | None:
    """Validate an optional weather source type."""
    if source_type is None:
        return None

    if not isinstance(source_type, str):
        raise ValueError(
            "source_type must be one of: "
            "alert, forecast, observation"
        )

    source_type = source_type.strip().lower()

    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            "source_type must be one of: "
            "alert, forecast, observation"
        )

    return source_type


def _perform_search(
    query: Any,
    top_k: Any,
    source_type: Any,
) -> dict[str, Any]:
    """
    Validate and execute a semantic weather search.

    Raises:
        ValueError: For invalid request parameters.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")

    source_type = _validate_source_type(
        source_type
    )

    top_k = _parse_positive_int(
        top_k,
        default=DEFAULT_TOP_K,
        maximum=MAX_TOP_K,
    )

    query = query.strip()

    query_embedding = embed_query(query)

    rows = search_weather_documents(
        query_embedding=query_embedding,
        top_k=top_k,
        source_type=source_type,
    )

    results = [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "location": row["location"],
            "headline": row["headline"],
            "chunk_text": row["chunk_text"],
            "source_type": row["source_type"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Application initialization
# ---------------------------------------------------------------------------

def initialize_application() -> None:
    """Initialize application dependencies once per process."""
    logger.info(
        "Initializing Weather Service application"
    )

    ensure_weather_schema()

    # Load the model during startup rather than on the first
    # search request. This moves the model-loading cost out of
    # the request path.
    load_embedding_model()

    logger.info(
        "Weather Service initialization complete"
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    """Application liveness check."""
    return jsonify(
        {
            "status": "ok",
            "service": "weather-app",
        }
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the weather application UI."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Weather synchronization
# ---------------------------------------------------------------------------

@app.post("/weather/sync")
def sync_weather():
    """Synchronize weather data from the National Weather Service."""
    body = request.get_json(
        silent=True
    )

    if body is None:
        body = {}

    if not isinstance(body, dict):
        return jsonify(
            {
                "error": "JSON object required"
            }
        ), 400

    locations = body.get("locations")

    if locations is None:
        locations = _load_default_locations()

    if not isinstance(locations, list):
        return jsonify(
            {
                "error": "locations must be a list"
            }
        ), 400

    if not locations:
        return jsonify(
            {
                "error": "locations cannot be empty"
            }
        ), 400

    limit = _parse_positive_int(
        body.get(
            "limit",
            DEFAULT_SYNC_LIMIT,
        ),
        default=DEFAULT_SYNC_LIMIT,
        maximum=MAX_SYNC_LIMIT,
    )

    logger.info(
        "Synchronizing weather for %d locations",
        len(locations),
    )

    try:
        synced = sync_weather_documents(
            locations=locations,
            limit=limit,
        )
        logger.info(f"Sync completed, synced={synced} (type={type(synced).__name__})")
    except Exception:
        logger.exception(
            "Weather synchronization failed with exception"
        )

        return jsonify(
            {
                "error": "Weather synchronization failed",
                "synced": 0,
                "locations": locations,
            }
        ), 500

    # Defensive validation: ensure synced is a valid integer
    if synced is None:
        logger.error("sync_weather_documents returned None!")
        synced = 0
    elif not isinstance(synced, int):
        logger.error(f"sync_weather_documents returned non-integer: {type(synced).__name__}")
        synced = 0

    return jsonify(
        {
            "synced": synced,
            "locations": locations,
        }
    )


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

@app.post("/weather/search")
def weather_search_post():
    """
    Semantic search over weather embeddings.

    Example request:

        {
            "query": "risk of flooding near rivers",
            "top_k": 5,
            "source_type": "alert"
        }
    """
    if not request.is_json:
        return jsonify(
            {
                "error": "JSON body required"
            }
        ), 400

    body = request.get_json(
        silent=True
    )

    if not isinstance(body, dict):
        return jsonify(
            {
                "error": "JSON object required"
            }
        ), 400

    try:
        response = _perform_search(
            query=body.get("query"),
            top_k=body.get(
                "top_k",
                DEFAULT_TOP_K,
            ),
            source_type=body.get(
                "source_type"
            ),
        )
    except ValueError as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400
    except Exception:
        logger.exception(
            "Weather semantic search failed"
        )

        return jsonify(
            {
                "error": "Search failed"
            }
        ), 500

    return jsonify(response)


@app.get("/weather/search")
def weather_search_get():
    """
    Convenience GET interface for semantic search.

    Example:

        /weather/search?query=flooding&top_k=5&source_type=alert
    """
    try:
        response = _perform_search(
            query=request.args.get("query"),
            top_k=request.args.get(
                "top_k",
                DEFAULT_TOP_K,
            ),
            source_type=request.args.get(
                "source_type"
            ),
        )
    except ValueError as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400
    except Exception:
        logger.exception(
            "Weather semantic search failed"
        )

        return jsonify(
            {
                "error": "Search failed"
            }
        ), 500

    return jsonify(response)


# ---------------------------------------------------------------------------
# Weather documents
# ---------------------------------------------------------------------------

@app.get("/weather/documents")
def weather_documents():
    """Return recently synchronized weather documents."""
    limit = _parse_positive_int(
        request.args.get(
            "limit",
            DEFAULT_DOCUMENT_LIMIT,
        ),
        default=DEFAULT_DOCUMENT_LIMIT,
        maximum=MAX_DOCUMENT_LIMIT,
    )

    try:
        rows = lakebase.run_query(
            f"""
            SELECT
                id,
                location,
                source_type,
                headline,
                narrative_text,
                issued_at,
                effective_at,
                synced_at
            FROM {WEATHER_DOCUMENTS_TABLE}
            ORDER BY synced_at DESC
            LIMIT %s
            """,
            (limit,),
        )
    except Exception:
        logger.exception(
            "Failed to retrieve weather documents"
        )

        return jsonify(
            {
                "error": "Failed to retrieve documents"
            }
        ), 500

    return jsonify(rows)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors without logging a full stack trace."""
    logger.info(
        "404 Not Found: %s %s",
        request.method,
        request.path,
    )
    return jsonify(
        {
            "error": "Not found"
        }
    ), 404


@app.errorhandler(Exception)
def handle_exception(
    error: Exception,
):
    """
    Return a consistent response for unexpected errors.

    Internal exception details are logged but not returned to
    clients.
    """
    # Don't log 404s as exceptions - they're handled above
    status = getattr(
        error,
        "code",
        500,
    )

    if not isinstance(status, int):
        status = 500

    if status != 404:
        logger.exception(
            "Unhandled application error"
        )

    return jsonify(
        {
            "error": "Internal server error"
        }
    ), status


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

initialize_application()


if __name__ == "__main__":
    host = os.getenv(
        "FLASK_RUN_HOST",
        "0.0.0.0",
    )

    port = _parse_positive_int(
        os.getenv(
            "FLASK_RUN_PORT",
            8000,
        ),
        default=8000,
        maximum=65535,
    )

    app.run(
        host=host,
        port=port,
        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "false",
            ).lower()
            == "true"
        ),
    )

