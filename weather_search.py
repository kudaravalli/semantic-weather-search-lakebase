"""
Weather semantic search utilities.

Uses:
    sentence-transformers/all-MiniLM-L6-v2

Embedding dimension:
    384

Database:
    Lakebase PostgreSQL + pgvector

Search:
    Cosine similarity using pgvector's <=> operator.

Index:
    HNSW with vector_cosine_ops.
"""

from __future__ import annotations

import logging
from typing import Any

import lakebase
from sentence_transformers import SentenceTransformer

from weather_schema import (
    EMBEDDING_DIMENSION,
    WEATHER_EMBEDDINGS_TABLE,
)


logger = logging.getLogger("weather-search")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOP_K = 20

VALID_SOURCE_TYPES = frozenset(
    {
        "alert",
        "forecast",
        "observation",
    }
)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.

    The model is initialized once per application process and reused
    for subsequent embedding requests.
    """
    global _MODEL

    if _MODEL is None:
        logger.info(
            "Loading embedding model: %s",
            MODEL_NAME,
        )

        _MODEL = SentenceTransformer(
            MODEL_NAME
        )

        logger.info(
            "Embedding model loaded successfully"
        )

    return _MODEL


def get_embedding_model() -> SentenceTransformer:
    """Return the cached embedding model."""
    return load_embedding_model()


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def embed_query(
    query: str,
) -> list[float]:
    """
    Generate a normalized embedding for text.

    Normalized embeddings are compatible with cosine similarity
    searches using pgvector's <=> operator.
    """
    if not isinstance(query, str):
        raise ValueError(
            "query must be a string"
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query must be a non-empty string"
        )

    model = get_embedding_model()

    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if vector.shape[0] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension "
            f"{EMBEDDING_DIMENSION}, "
            f"got {vector.shape[0]}"
        )

    return vector.tolist()


# ---------------------------------------------------------------------------
# Vector formatting
# ---------------------------------------------------------------------------

def _vector_to_pgvector(
    vector: list[float],
) -> str:
    """
    Convert an embedding to pgvector's textual representation.

    Example:
        [0.1,0.2,0.3]
    """
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension "
            f"{EMBEDDING_DIMENSION}, "
            f"got {len(vector)}"
        )

    try:
        return (
            "["
            + ",".join(
                str(float(value))
                for value in vector
            )
            + "]"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding contains a non-numeric value"
        ) from exc


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def search_weather_documents(
    query_embedding: list[float],
    top_k: int = 5,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search weather documents using cosine similarity.

    pgvector's <=> operator returns cosine distance:

        0 = identical
        1 = increasingly dissimilar

    Therefore similarity is calculated as:

        1 - cosine_distance

    Args:
        query_embedding:
            A 384-dimensional normalized embedding.

        top_k:
            Maximum number of results to return.

        source_type:
            Optional source filter:
                alert
                forecast
                observation

    Returns:
        Search results ordered by descending cosine similarity.
    """
    if not isinstance(query_embedding, list):
        raise ValueError(
            "query_embedding must be a list"
        )

    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension "
            f"{EMBEDDING_DIMENSION}, "
            f"got {len(query_embedding)}"
        )

    try:
        top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "top_k must be an integer"
        ) from exc

    top_k = max(
        1,
        min(top_k, MAX_TOP_K),
    )

    if source_type is not None:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                "source_type must be one of: "
                + ", ".join(
                    sorted(VALID_SOURCE_TYPES)
                )
            )

    vector_string = _vector_to_pgvector(
        query_embedding
    )

    # Keep the vector expression directly in the ORDER BY clause.
    #
    # This is intentional: pgvector's HNSW index is designed around
    # nearest-neighbor ORDER BY operations such as:
    #
    #     embedding <=> query_vector
    #
    # The vector_cosine_ops HNSW index created by weather_schema.py
    # matches this operator.

    if source_type is None:
        sql = f"""
            SELECT
                e.id,
                e.document_id,
                e.location,
                e.source_type,
                e.headline,
                e.chunk_text,

                1 - (
                    e.embedding <=> %s::vector
                ) AS similarity

            FROM {WEATHER_EMBEDDINGS_TABLE} AS e

            ORDER BY
                e.embedding <=> %s::vector

            LIMIT %s
        """

        params = (
            vector_string,
            vector_string,
            top_k,
        )

    else:
        sql = f"""
            SELECT
                e.id,
                e.document_id,
                e.location,
                e.source_type,
                e.headline,
                e.chunk_text,

                1 - (
                    e.embedding <=> %s::vector
                ) AS similarity

            FROM {WEATHER_EMBEDDINGS_TABLE} AS e

            WHERE e.source_type = %s

            ORDER BY
                e.embedding <=> %s::vector

            LIMIT %s
        """

        params = (
            vector_string,
            source_type,
            vector_string,
            top_k,
        )

    with lakebase.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                params,
            )

            rows = cursor.fetchall()

    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "location": row["location"],
            "source_type": row["source_type"],
            "headline": row["headline"],
            "chunk_text": row["chunk_text"],
            "similarity": float(
                row["similarity"]
            ),
        }
        for row in rows
    ]

