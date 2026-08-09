"""
Databricks notebook:
ingest_weather_embeddings.py

Purpose:

Generate vector embeddings for weather documents
stored in Lakebase using psycopg and pgvector.

This notebook is called by:

resources/ingest_weather_embeddings_job.yml


Flow:

Notebook
   |
   v
embedding/weather_embedding_pipeline.py
   |
   +--> Read unembedded documents
   |
   +--> Chunk text (sliding window)
   |
   +--> Load SentenceTransformer model
   |
   +--> Generate embeddings
   |
   +--> Upsert to Lakebase

"""

from __future__ import annotations


import json
import logging
import os
import sys



# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)


logger = logging.getLogger(
    "ingest-weather-embeddings"
)



# ----------------------------------------------------------------------
# Make project modules available
#
# When deployed with Databricks Asset Bundles,
# the repo root is mounted with the notebook.
# ----------------------------------------------------------------------

sys.path.append(
    os.path.abspath("..")
)



import lakebase


from weather_schema import (
    ensure_weather_schema,
)


from embedding.weather_embedding_pipeline import (
    run as run_embedding_pipeline,
)





# ----------------------------------------------------------------------
# Job parameters
# ----------------------------------------------------------------------

try:

    dbutils

except NameError:

    dbutils = None





def get_parameter(
    name: str,
    default: str,
):

    if dbutils:

        try:

            return dbutils.widgets.get(
                name
            )

        except Exception:

            pass


    return default





weather_documents_table = get_parameter(
    "weather_documents_table",
    "weather.weather_documents",
)



weather_embeddings_table = get_parameter(
    "weather_embeddings_table",
    "weather.weather_embeddings",
)





# ----------------------------------------------------------------------
# Initialize schema
# ----------------------------------------------------------------------

ensure_weather_schema()



logger.info(
    "Source table: %s",
    weather_documents_table,
)


logger.info(
    "Target table: %s",
    weather_embeddings_table,
)




# ----------------------------------------------------------------------
# Run embedding pipeline
#
# The pipeline function internally:
#
# - reads unembedded documents
# - chunks text using sliding window
# - loads the SentenceTransformer model
# - generates embeddings in batches
# - upserts to Lakebase
#
# ----------------------------------------------------------------------

embeddings_count = run_embedding_pipeline()




# ----------------------------------------------------------------------
# Output metrics
# ----------------------------------------------------------------------

result = {

    "weather_documents_table":
        weather_documents_table,


    "weather_embeddings_table":
        weather_embeddings_table,


    "embeddings_generated":
        embeddings_count,

}



logger.info(
    "Embedding ingestion complete: %s",
    result,
)



print(
    json.dumps(
        result,
        indent=2,
    )
)

