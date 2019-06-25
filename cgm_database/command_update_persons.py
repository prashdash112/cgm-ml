import warnings
warnings.filterwarnings("ignore")
import dbutils
import glob2 as glob
import os
import progressbar
import pandas as pd
import sys
import time


whhdata_path = "/whhdata"


def execute_command_updatemeasurements():
    print("Updating persons...")
    
    main_connector = dbutils.connect_to_main_database()
    
    # TODO import persons
    
    # Where to get the data.
    glob_search_path = os.path.join(whhdata_path, "*.csv")
    csv_paths = sorted(glob.glob(glob_search_path))
    csv_paths.sort(key=os.path.getmtime)
    csv_path = csv_paths[-1]
    print("Using {}".format(csv_path))

    # Load the data-frame.
    df = pd.read_csv(csv_path)
    
    # List all columns.
    columns = list(df)
    print(columns)
    
#    ['personId', 'qrcode', 'sex', 'type', 'age', 'height', 'weight', 'muac', 'headCircumference', 'oedema', 'latitude', 'longitude', 'address', 'timestamp', 'deleted', 'deletedBy', 'visible', 'createdBy']
    
    """
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
    """
    
    table = "person"

    # Number of rows before.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows before: {}".format(rows_number))

    # Insert data in batches.
    batch_size = 1000
    sql_statement = ""
    rows_number_df = len(df.index)
    bar = progressbar.ProgressBar(max_value=rows_number_df)
    used_ids = []
    for index, row in df.iterrows():
        bar.update(index)
        
        # Make sure there are no duplicates. Local.
        select_sql_statement = "SELECT COUNT(*) FROM person WHERE id='{}'".format(row["personId"])
        result = main_connector.execute(select_sql_statement, fetch_one=True)[0]
        if row["personId"] in used_ids or result != 0:
            #print(row["personId"], "already in DB")
            pass
        else:
        
            # TODO check all of these.
            insert_data = {}
            insert_data["id"] = row["personId"]
            insert_data["name"] = "UNKNOWN" 
            insert_data["surname"] = "UNKNOWN" 
            insert_data["birthday"] = 0   
            insert_data["sex"] = "UNKNOWN" 
            insert_data["guardian"] = "UNKNOWN" 
            insert_data["is_age_estimated"] = False
            insert_data["qr_code"] = row["qrcode"]    
            insert_data["created"] = 0
            insert_data["timestamp"] = row["timestamp"]
            insert_data["created_by"] = row["createdBy"]
            insert_data["deleted"] = row["deleted"]
            insert_data["deleted_by"] = row["deletedBy"]

            sql_statement += dbutils.create_insert_statement(table, insert_data.keys(), insert_data.values())
            #print(sql_statement)
            used_ids.append(row["personId"])
        
        if index != 0 and ((index % batch_size) == 0 or index == rows_number_df - 1) and sql_statement != "":
            main_connector.execute(sql_statement)
            sql_statement = ""

    bar.finish()

    # Number of rows after sync.
    rows_number = main_connector.get_number_of_rows(table)
    print("Number of rows after: {}".format(rows_number))
    
    
if __name__ == "__main__":
    execute_command_updatemeasurements()