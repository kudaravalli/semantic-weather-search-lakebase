-- 05_seed_default_locations.sql
--
-- Seed default application locations into weather.location_cache.
--
-- This file is safe to run multiple times.
-- NWS grid metadata is intentionally left NULL because it should be
-- populated by the application's NWS lookup when the location is used.

INSERT INTO weather.location_cache
(
    location_key,
    latitude,
    longitude,
    display_name,
    nws_office,
    nws_grid_x,
    nws_grid_y,
    payload,
    created_at,
    updated_at
)
VALUES
(
    'chicago-il',
    41.8781,
    -87.6298,
    'Chicago, IL',
    NULL,
    NULL,
    NULL,
    '{}'::jsonb,
    NOW(),
    NOW()
),
(
    'austin-tx',
    30.2672,
    -97.7431,
    'Austin, TX',
    NULL,
    NULL,
    NULL,
    '{}'::jsonb,
    NOW(),
    NOW()
),
(
    'san-francisco-ca',
    37.7749,
    -122.4194,
    'San Francisco, CA',
    NULL,
    NULL,
    NULL,
    '{}'::jsonb,
    NOW(),
    NOW()
)
ON CONFLICT (location_key)
DO UPDATE SET
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    display_name = EXCLUDED.display_name,
    updated_at = NOW();

