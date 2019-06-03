# Database.

This folder contains means and methods to access the CGM database. 

Things you need to know:
- This module contains a CLI (cli.py) for interacting with the database during productions. Have a look at it to get an idea about what is going on.
- There is also a schema for our database (schema.sql). It creates all our necessary tables. Can be executed via the CLI.
- Every user has to create their own credentials file (dbconnection.json). This is deliberately not maintained via git for security reasons. The file is supposed to look like this:
```
{
    "dbname": "cgm_ml_db",
    "user": INSERT_USER_NAME_HERE,
    "host": "cgm-dev.postgres.database.azure.com",
    "password": INSERT_PASSWORD-ERE,
    "port": 5432,
    "sslmode": "require"
}
```
- There are a couple of notebooks available that will show you basic functionality.