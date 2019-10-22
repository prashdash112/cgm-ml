DROP TABLE IF EXISTS person CASCADE;
DROP TABLE IF EXISTS measure CASCADE;
DROP TABLE IF EXISTS artifact CASCADE;
DROP TABLE IF EXISTS artifact_result CASCADE;
DROP TABLE IF EXISTS measure_result CASCADE;
DROP TABLE IF EXISTS device CASCADE;
DROP TABLE IF EXISTS model CASCADE;

CREATE TABLE IF NOT EXISTS model ( 
	id                   varchar(255)  NOT NULL ,
	name                 varchar(1000)  NOT NULL ,
	"version"            varchar(255)  NOT NULL ,
	json_metadata        json   ,
	CONSTRAINT pk_metadata_id PRIMARY KEY ( id )
 );
 
-- Creates a table for person
CREATE TABLE IF NOT EXISTS person (
	id                   varchar(255) NOT NULL ,
	name                 text  NOT NULL ,
	surname              text  NOT NULL ,
	birthday             bigint  NOT NULL ,
	sex                  text  NOT NULL ,
	guardian             text  NOT NULL ,
	is_age_estimated     bool  NOT NULL ,
	qr_code              text  NOT NULL ,
	create_timestamp     bigint  NOT NULL ,
	sync_timestamp       bigint  NOT NULL ,
    created_by           text  NOT NULL ,
	deleted              bool  NOT NULL ,
	deleted_by           text  NOT NULL, 
    CONSTRAINT person_pkey PRIMARY KEY ( id )
 );

-- Creates a table for measure
CREATE TABLE IF NOT EXISTS measure ( 
	id                   varchar(255)  NOT NULL ,
	person_id            text NOT NULL ,
    -- date -> create_timestamp
	"type"               text  NOT NULL ,
	qr_code              varchar(255)   ,
    age                  bigint  NOT NULL ,
	height               real ,
	weight               real ,
	muac                 real ,
	head_circumference   real ,
	longitude            float8 ,
    latitude             float8 ,
    address              text ,
    visible              bool  NOT NULL ,
	oedema               bool  NOT NULL ,
	created_by           text  NOT NULL ,
	deleted              bool  NOT NULL ,
	deleted_by           text ,
	create_timestamp     bigint  NOT NULL ,
	sync_timestamp       bigint  NOT NULL ,
	CONSTRAINT measure_pkey PRIMARY KEY ( id ),
	CONSTRAINT measure_person_id_fkey FOREIGN KEY ( person_id ) REFERENCES person( id )  
 );

-- Creates a table for artifact
CREATE TABLE IF NOT EXISTS artifact (
	id                   varchar(255)  NOT NULL ,
	measure_id           text  NOT NULL ,
	dataformat               text  NOT NULL , -- this column was "type"
	storage_path               text  NOT NULL ,
    scan_step            integer NOT NULL ,
	hash_value           text  NOT NULL ,
	file_size            bigint  NOT NULL ,
	deleted              bool  NOT NULL ,
	qr_code              text  NOT NULL ,
	age                  bigint   ,
    created_by           text  NOT NULL ,
	status               integer  NOT NULL ,
	upload_timestamp     bigint  NOT NULL ,
	create_timestamp     bigint  NOT NULL ,
	sync_timestamp       real   ,
	CONSTRAINT artifact_pkey PRIMARY KEY ( id ),
	CONSTRAINT artifact_measure_id_fkey FOREIGN KEY ( measure_id ) REFERENCES measure( id )  
 );

CREATE TABLE  IF NOT EXISTS artifact_result ( 
	artifact_id          varchar(255)  NOT NULL ,
	model_id             text  NOT NULL ,
	"key"                text  NOT NULL ,
    float_value          float8   ,
    json_value           json   ,
    confidence_value     float8   ,
    CONSTRAINT artifact_result_pkey PRIMARY KEY ( artifact_id, model_id, "key" ),
	CONSTRAINT artifact_result_artifact_id_fkey FOREIGN KEY ( artifact_id ) REFERENCES artifact( id ) ,
    CONSTRAINT fk_artifact_result_model FOREIGN KEY ( model_id ) REFERENCES model( id )    
 );


CREATE TABLE IF NOT EXISTS measure_result ( 
	measure_id           varchar(255)  NOT NULL ,
	model_id             text  NOT NULL ,
	"key"                text  NOT NULL ,
	float_value          float8   ,
	json_value           json   ,
	confidence_value     float8   ,
	CONSTRAINT measure_result_pkey PRIMARY KEY ( measure_id, model_id, "key" ),
	CONSTRAINT measure_result_measure_id_fkey FOREIGN KEY ( measure_id ) REFERENCES measure( id )  ,
	CONSTRAINT fk_measure_result_model FOREIGN KEY ( model_id ) REFERENCES model( id )  
 );


CREATE TABLE IF NOT EXISTS device ( 
	id                   varchar(255)  NOT NULL ,
	uuid                 varchar(100)   ,
	create_timestamp     real   ,
	sync_timestamp       real   ,
	new_artifact_file_size_mb real   ,
	new_artifacts        numeric   ,
	deleted_artifacts    numeric   ,
	total_artifact_file_size_mb real   ,
	total_artifacts      numeric   ,
	own_measures         numeric   ,
	own_persons          numeric   ,
	created_by           varchar(1000)   ,
	total_measures       numeric   ,
	total_persons        numeric   ,
    app_version          varchar(50) ,
	CONSTRAINT pk_device_id PRIMARY KEY ( id )
 );

 -- TODO result_transaction table needed if results are produced in the app backend?