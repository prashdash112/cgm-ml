import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json

def connect_to_default_database():
    json_data = load_dbconnection_file()
    connection = psycopg2.connect(
        dbname="postgres",
        user=json_data["user"],
        host=json_data["host"],
        password=json_data["password"],
        port=json_data["port"],
        sslmode=json_data["sslmode"]
    )
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return connection
    
    
def connect_to_main_database():
    json_data = load_dbconnection_file()
    connection = psycopg2.connect(
        dbname=json_data["dbname"],
        user=json_data["user"],
        host=json_data["host"],
        password=json_data["password"],
        port=json_data["port"],
        sslmode=json_data["sslmode"]
    )
    return connection
    
    
def load_dbconnection_file():
    with open("dbconnection.json") as json_file:  
        json_data = json.load(json_file)
        return json_data

