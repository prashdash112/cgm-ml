import warnings
warnings.filterwarnings("ignore")
import dbutils

main_connector = dbutils.connect_to_main_database()

def execute_command_statistics():
    result_string = ""
    
    # Getting table sizes.
    tables = ["person", "measure", "artifact", "artifact_quality"]
    for table in tables:
        sql_statement = "SELECT COUNT(*) FROM {};".format(table)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table '{}' has {} entries.\n".format(table, result)
    
    # Find the number of rows that lack measurement-id.
    sql_statement = "SELECT COUNT(*) FROM artifact WHERE measure_id IS NULL;"
    result = main_connector.execute(sql_statement, fetch_one=True)[0]
    result_string += "Table 'artifact' has {} entries without measure-id.\n".format(result)

    artifact_types = ["pcd", "rgb"]
    for artifact_type in artifact_types:
        sql_statement = "SELECT COUNT(*) FROM artifact WHERE type='{}';".format(artifact_type)
        result = main_connector.execute(sql_statement, fetch_one=True)[0]
        result_string += "Table 'artifact' has {} entries with type '{}'.\n".format(result, artifact_type)


    print(result_string)

    
if __name__ == "__main__":
    execute_command_statistics()
   