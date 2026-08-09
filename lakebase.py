"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single Lakebase connection URL stored in a Databricks
secret scope. The URL should be a standard PostgreSQL connection URL, e.g.:

    postgresql://role:password@host:5432/databricks_postgres?sslmode=require

The URL is base64-encoded in the Databricks secret.

Uses Psycopg 3 for PostgreSQL connectivity.
"""

import base64
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.rows import dict_row
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_w = WorkspaceClient()

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _lakebase_url() -> str:
    """Fetch and decode the Lakebase connection URL from the Databricks secret."""
    secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a raw Psycopg 3 connection with dictionary rows.

    Rows returned by cursors are dictionaries, e.g.:

        {"city": "San Francisco", "temperature": 65}
    """
    conn = psycopg.connect(
        _lakebase_url(),
        row_factory=dict_row,
    )

    try:
        yield conn
    finally:
        conn.close()


def get_engine() -> Engine:
    """
    Return a SQLAlchemy engine configured to use Psycopg 3.
    """
    url = _lakebase_url()

    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]

    return create_engine(url)


def run_query(
    sql: str,
    params: tuple | dict | None = None,
) -> list[dict]:
    """
    Run a read query against Lakebase.

    Returns:
        List of rows represented as dictionaries.

    Example:
        rows = run_query(
            "SELECT * FROM weather WHERE city = %s",
            ("San Francisco",),
        )
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(
    sql: str,
    params: tuple | dict | None = None,
) -> int:
    """
    Run an INSERT/UPDATE/DELETE against Lakebase.

    Returns:
        Number of rows affected.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            affected_rows = cur.rowcount
            conn.commit()
            return affected_rows

