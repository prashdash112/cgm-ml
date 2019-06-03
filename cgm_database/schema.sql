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
    weight_kgs REAL NOT NULL,
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
    last_updated REAL NOT NULL,
    rejected_by_expert BOOLEAN NOT NULL,
    had_error BOOLEAN NOT NULL,
    error_message TEXT,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    blur_variance REAL NOT NULL,
    has_face BOOLEAN NOT NULL,
    measurement_id INTEGER REFERENCES measurements(id)
);

-- Creates a table for pointcloud data.
CREATE TABLE IF NOT EXISTS pointcloud_data (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,
    qrcode TEXT NOT NULL,
    last_updated REAL NOT NULL,
    rejected_by_expert BOOLEAN NOT NULL,
    had_error BOOLEAN NOT NULL,
    error_message TEXT,
    number_of_points INTEGER NOT NULL,  
    confidence_min REAL NOT NULL,
    confidence_avg REAL NOT NULL,
    confidence_std REAL NOT NULL,
    confidence_max REAL NOT NULL,
    centroid_x REAL NOT NULL, 
    centroid_y REAL NOT NULL, 
    centroid_z REAL NOT NULL, 
    stdev_x REAL NOT NULL,
    stdev_y REAL NOT NULL,
    stdev_z REAL NOT NULL,

    measurement_id INTEGER REFERENCES measurements(id)
);

-- Creates a table for artifacts
CREATE TABLE IF NOT EXISTS artifacts (
    id VARCHAR(255) PRIMARY KEY,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    hash_value TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    upload_date BIGINT NOT NULL,
    deleted BOOLEAN NOT NULL,
    qr_code TEXT NOT NULL,
    create_date BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    status integer NOT NULL
);

-- Creates a table for persons
CREATE TABLE IF NOT EXISTS persons (
    id VARCHAR(255) PRIMARY KEY,
    name TEXT NOT NULL,
    surname TEXT NOT NULL,
    birthday BIGINT NOT NULL,
    sex TEXT NOT NULL,
    guardian TEXT NOT NULL,
    is_age_estimated BOOLEAN NOT NULL,
    qr_code TEXT NOT NULL,
    created BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    deleted BOOLEAN NOT NULL,
    deleted_by TEXT NOT NULL
);


-- Creates a table for measures
CREATE TABLE IF NOT EXISTS measures (
    id VARCHAR(255) PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    date BIGINT NOT NULL,
    type TEXT NOT NULL,
    age BIGINT NOT NULL,
    height DOUBLE NOT NULL,
    weight DOUBLE NOT NULL,
    muac DOUBLE NOT NULL,
    head_circumference DOUBLE NOT NULL,
    artifact TEXT NOT NULL,
    visible BOOLEAN NOT NULL,
    oedema BOOLEAN NOT NULL,
    timestamp BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    deleted BOOLEAN NOT NULL,
    deleted_by TEXT NOT NULL    
);