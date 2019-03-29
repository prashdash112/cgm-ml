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
    connection.autocommit = True
    return DatabaseInterface(connection)
    
    
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
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    connection.autocommit = True
    return DatabaseInterface(connection)
    
    
def load_dbconnection_file():
    with open("dbconnection.json") as json_file:  
        json_data = json.load(json_file)
        return json_data

    
class DatabaseInterface:
    
    def __init__(self, connection):
        self.connection = connection
        self.cursor = connection.cursor()
        
    def execute(self, script):
        self.cursor.execute(script)
        self.connection.commit()
        
    def execute_script_file(self, filename):
        self.cursor.execute(open(filename, "r").read())
        self.connection.commit()
    
    def get_all_tables(self):
        self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [str(table[0]) for table in self.cursor.fetchall()]
        return tables
    
    def clear_table(self, table):
        result = self.cursor.execute("TRUNCATE TABLE {};".format(table))
        self.connection.commit()

    def get_number_of_rows(self, table):
        self.cursor.execute("SELECT COUNT(*) from {};".format(table))
        result = self.cursor.fetchone()
        return result[0]
    

def create_insert_statement(keys, values):
    sql_statement = "INSERT INTO {}".format("measurements") + " "
    
    keys_string = "(" + ", ".join(keys) + ")"
    sql_statement += keys_string

    values_string = "VALUES (" + ", ".join(values) + ")"
    sql_statement += "\n" + values_string
    
    sql_statement += ";" + "\n"
    
    return sql_statement