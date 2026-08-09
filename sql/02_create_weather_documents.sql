CREATE TABLE IF NOT EXISTS weather.weather_documents
(
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
);

COMMENT ON TABLE weather.weather_documents IS
'Normalized NWS weather alerts and forecast documents';

