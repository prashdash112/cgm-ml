-- This removes everything. You better be careful.
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

-- Creates a table for measurements.
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

-- Creates a table for image data.
CREATE TABLE IF NOT EXISTS image_data (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,
    qrcode TEXT NOT NULL,
    targets TEXT NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    rejected_by_expert BOOLEAN NOT NULL,
    had_error BOOLEAN NOT NULL,
    error_message TEXT,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    blur_variance REAL NOT NULL
);

-- Creates a table for pointcloud data.
CREATE TABLE IF NOT EXISTS pointcloud_data (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,
    qrcode TEXT NOT NULL,
    targets TEXT NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    rejected_by_expert BOOLEAN NOT NULL,
    had_error BOOLEAN NOT NULL,
    error_message TEXT,
    number_of_points INTEGER NOT NULL,  
    confidence_min REAL NOT NULL,
    confidence_avg REAL NOT NULL,
    confidence_std REAL NOT NULL,
    confidence_max REAL NOT NULL
);