DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

CREATE TABLE IF NOT EXISTS measurements (
    id SERIAL PRIMARY KEY,
    measurement_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    qrcode TEXT NOT NULL,
    sex TEXT NOT NULL,
    type TEXT NOT NULL,
    age_days INTEGER NOT NULL,
    height_cms REAL NOT NULL,
    weight_cms REAL NOT NULL,
    muac_cms REAL NOT NULL,
    head_circumference_cms REAL NOT NULL,
    oedema BOOLEAN NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT NOT NULL,
    timestamp DECIMAL NOT NULL,
    deleted BOOLEAN NOT NULL,
    deleted_by TEXT NOT NULL,
    visible BOOLEAN NOT NULL,
    created_by TEXT NOT NULL
);

