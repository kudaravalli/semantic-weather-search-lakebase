"""
Lakebase schema definitions for the weather application.

Creates:

- weather_documents
    Normalized NWS weather documents.

- location_cache
    Cached location -> coordinates -> NWS grid resolution.

- weather_embeddings
    Canonical embedding store for semantic weather search.

Embeddings use pgvector with dimension 384, matching:
sentence-transformers/all-MiniLM-L6-v2.

Semantic search uses pgvector cosine distance (<=>) with an HNSW
vector index.
"""

from __future__ import annotations

import lakebase


# ---------------------------------------------------------------------------
# Schema names
# ---------------------------------------------------------------------------

WEATHER_SCHEMA = "weather"

WEATHER_DOCUMENTS_TABLE = (
    f"{WEATHER_SCHEMA}.weather_documents"
)

LOCATION_CACHE_TABLE = (
    f"{WEATHER_SCHEMA}.location_cache"
)

WEATHER_EMBEDDINGS_TABLE = (
    f"{WEATHER_SCHEMA}.weather_embeddings"
)

EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------------------------
# Public initialization
# ---------------------------------------------------------------------------

def ensure_weather_schema() -> None:
    """
    Create the weather schema, tables, constraints, and indexes.

    All operations are idempotent and can safely be executed during
    application startup.
    """
    _create_schema()
    _create_extensions()
    _create_weather_documents_table()
    _create_location_cache_table()
    _create_embedding_table()
    _create_indexes()


# ---------------------------------------------------------------------------
# Schema and extensions
# ---------------------------------------------------------------------------

def _create_schema() -> None:
    """Create the application schema if it does not exist."""
    lakebase.run_write(
        f"""
        CREATE SCHEMA IF NOT EXISTS {WEATHER_SCHEMA}
        """
    )


def _create_extensions() -> None:
    """Enable pgvector."""
    lakebase.run_write(
        """
        CREATE EXTENSION IF NOT EXISTS vector
        """
    )


# ---------------------------------------------------------------------------
# Weather documents
# ---------------------------------------------------------------------------

def _create_weather_documents_table() -> None:
    """Create the normalized weather document table."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_DOCUMENTS_TABLE} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL
                            CHECK (
                              source_type IN (
                                'alert',
                                'forecast',
                                'observation'
                              )
                            ),
            headline TEXT,
            narrative_text TEXT,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            content_hash TEXT,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


# ---------------------------------------------------------------------------
# Location cache
# ---------------------------------------------------------------------------

def _create_location_cache_table() -> None:
    """Create the location-to-NWS-grid cache table."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {LOCATION_CACHE_TABLE} (
            location_key TEXT PRIMARY KEY,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            display_name TEXT,
            nws_office TEXT,
            nws_grid_x INTEGER,
            nws_grid_y INTEGER,
            payload JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _create_embedding_table() -> None:
    """
    Create the canonical weather embedding table.

    Each document has one deterministic embedding representing its
    searchable headline + narrative content.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_EMBEDDINGS_TABLE} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
                REFERENCES {WEATHER_DOCUMENTS_TABLE}(id)
                ON DELETE CASCADE,

            location TEXT,

            source_type TEXT
                CHECK (
                    source_type IS NULL
                    OR source_type IN (
                        'alert',
                        'forecast',
                        'observation'
                    )
                ),

            headline TEXT,

            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            content_hash TEXT,
            embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT fk_weather_embedding_document
                FOREIGN KEY(document_id)
                REFERENCES weather.weather_documents(id)
                ON DELETE CASCADE,

            CONSTRAINT uq_weather_embedding_chunk
                UNIQUE(document_id, chunk_index)
        )
        """
    )


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

def _create_indexes() -> None:
    """
    Create indexes used by document retrieval and semantic search.

    The HNSW index uses vector_cosine_ops because searches use the
    pgvector <=> cosine-distance operator.
    """

    def _safe_create_index(sql: str, index_name: str) -> None:
        """Create an index, ignoring permission errors if it already exists."""
        import logging
        logger = logging.getLogger(__name__)
    
        try:
            lakebase.run_write(sql)
        except Exception as e:
            if type(e).__name__ == "InsufficientPrivilege":
                logger.warning(f"Skipping {index_name}: insufficient privileges (table owned by different role)")
            else:
                raise

    # -----------------------------------------------------------------------
    # Weather document indexes
    # -----------------------------------------------------------------------
    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_documents_source_type_idx
        ON {WEATHER_DOCUMENTS_TABLE} (source_type)
        """,
        "weather_documents_source_type_idx"
    )

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_documents_synced_at_idx
        ON {WEATHER_DOCUMENTS_TABLE} (synced_at DESC)
        """,
        "weather_documents_synced_at_idx"
    )

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_documents_location_idx
        ON {WEATHER_DOCUMENTS_TABLE} (location)
        """,
        "weather_documents_location_idx"
    )

    # -----------------------------------------------------------------------
    # Location cache indexes
    # -----------------------------------------------------------------------

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        location_cache_updated_at_idx
        ON {LOCATION_CACHE_TABLE} (updated_at DESC)
        """,
        "location_cache_updated_at_idx"
    )

    # -----------------------------------------------------------------------
    # Embedding indexes
    # -----------------------------------------------------------------------

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_embeddings_document_id_idx
        ON {WEATHER_EMBEDDINGS_TABLE} (document_id)
        """,
        "weather_embeddings_document_id_idx"
    )

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_embeddings_source_type_idx
        ON {WEATHER_EMBEDDINGS_TABLE} (source_type)
        """,
        "weather_embeddings_source_type_idx"
    )

    # -----------------------------------------------------------------------
    # HNSW vector index
    # -----------------------------------------------------------------------
    #
    # <=> = cosine distance
    # vector_cosine_ops = cosine operator class
    #
    # HNSW is preferred here because it provides strong approximate
    # nearest-neighbor performance without requiring periodic IVFFlat
    # list tuning/rebuilding as the dataset grows.

    _safe_create_index(
        f"""
        CREATE INDEX IF NOT EXISTS
        weather_embeddings_embedding_hnsw_idx
        ON {WEATHER_EMBEDDINGS_TABLE}
        USING hnsw (embedding vector_cosine_ops)
        """,
        "weather_embeddings_embedding_hnsw_idx"
    )

