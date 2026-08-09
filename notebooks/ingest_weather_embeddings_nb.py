# Databricks notebook source
# DBTITLE 1,Ingest Weather Embeddings
# Databricks notebook - Ingest Weather Embeddings

from __future__ import annotations
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-weather-embeddings")

# Make project modules available
# Find the bundle files directory by looking for it in sys.path
bundle_files_dir = None
for path in sys.path:
    if path.endswith('/notebooks'):
        # Found the notebooks directory, parent is the files directory
        bundle_files_dir = os.path.dirname(path)
        break

if bundle_files_dir is None:
    # Fallback: try to construct the path from the notebook path
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    # Ensure it has /Workspace prefix
    if not notebook_path.startswith('/Workspace/'):
        notebook_path = '/Workspace' + notebook_path
    bundle_files_dir = os.path.dirname(os.path.dirname(notebook_path))

logger.info(f"Bundle files directory: {bundle_files_dir}")

# Add to sys.path if not already there
if bundle_files_dir and bundle_files_dir not in sys.path:
    sys.path.insert(0, bundle_files_dir)
    logger.info(f"Added to sys.path: {bundle_files_dir}")

# Import project modules with error handling
try:
    import lakebase
    from weather_schema import ensure_weather_schema
    from embedding.weather_embedding_pipeline import run as run_embedding_pipeline
    logger.info("Successfully imported project modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    if bundle_files_dir and os.path.exists(bundle_files_dir):
        logger.error(f"Files in bundle_files_dir ({bundle_files_dir}): {os.listdir(bundle_files_dir)}")
    else:
        logger.error(f"Bundle files directory not found or is None: {bundle_files_dir}")
    raise

# Job parameters
try:
    dbutils
except NameError:
    dbutils = None

def get_parameter(name: str, default: str):
    if dbutils:
        try:
            return dbutils.widgets.get(name)
        except Exception:
            pass
    return default

weather_documents_table = get_parameter("weather_documents_table", "weather.weather_documents")
weather_embeddings_table = get_parameter("weather_embeddings_table", "weather.weather_embeddings")

logger.info("Source table: %s", weather_documents_table)
logger.info("Target table: %s", weather_embeddings_table)

# Initialize schema (ensures embeddings table exists)
ensure_weather_schema()

# Run embedding pipeline
embeddings_count = run_embedding_pipeline()

# Output metrics
result = {
    "weather_documents_table": weather_documents_table,
    "weather_embeddings_table": weather_embeddings_table,
    "embeddings_generated": embeddings_count,
}

logger.info("Embedding ingestion complete: %s", result)
print(json.dumps(result, indent=2))