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
        
    def execute(self, script, fetch_one=False, fetch_all=False):
        result = self.cursor.execute(script)
        self.connection.commit()
        if fetch_one == True:
            result = self.cursor.fetchone()
        elif fetch_all == True:
            result = self.cursor.fetchall()
        return result
        
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
    
    def get_columns(self, table):
        sql_statement = ""
        sql_statement += "SELECT column_name, data_type, character_maximum_length FROM INFORMATION_SCHEMA.COLUMNS" 
        sql_statement += " WHERE table_name = '{}';".format(table)
        self.cursor.execute(sql_statement)
        results = self.cursor.fetchall()
        columns = [result[0] for result in results]
        return columns
    

def create_insert_statement(table, keys, values, convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]
        
    sql_statement = "INSERT INTO {}".format(table) + " "
    
    keys_string = "(" + ", ".join(keys) + ")"
    sql_statement += keys_string
    
    values_string = "VALUES (" + ", ".join(values) + ")"
    sql_statement += "\n" + values_string
    
    sql_statement += ";" + "\n"
    
    return sql_statement


def create_update_statement(table, keys, values, id_value, convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]
        
    sql_statement = "UPDATE {}".format(table) + " SET"
    sql_statement += ", ".join([" {} = {}".format(key, value) for key, value in zip(keys, values) if key != id_value])
    sql_statement += " WHERE id = {}".format(id_value)
    sql_statement += ";" + "\n"

    return sql_statement


def create_select_statement(table, keys=[], values=[], convert_values_to_string=True, use_quotes_for_values=True):
    if convert_values_to_string == True:
        values = [str(value) for value in values]
    if use_quotes_for_values == True:
        values = ["'" + value + "'" for value in values]

    sql_statement = "SELECT * FROM {}".format(table)
    
    if len(keys) != 0 and len(values) != 0:
        sql_statement += " WHERE "
        like_statements = []
        for key, value in zip(keys, values):
            like_statement = str(key) + " LIKE " + str(value)
            like_statements.append(like_statement)
        sql_statement += " AND ".join(like_statements) 
    
    sql_statement += ";" + "\n"
    return sql_statement