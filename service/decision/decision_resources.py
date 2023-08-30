# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common resources used in the gRPC sight_service example."""

import json
import time
from google.cloud import spanner
OPERATION_TIMEOUT_SECONDS = 240


def write_to_JSON(new_log_entry):
    """Writes in the database locally as json file.

  Returns:
    
  """
    fname = "service/decision/decision_db.json"

    with open(fname, 'r+') as sight_service_db_file:
        # First we load existing data into a dict.
        log_data = json.load(sight_service_db_file)
        # Join new_data with log_data inside emp_details
        log_data["log_details"].append(new_log_entry)
        # Sets file's current position at offset.
        sight_service_db_file.seek(0)
        # convert back to json.
        json.dump(log_data, sight_service_db_file, indent = 4)


def create_database(instance_id, database_id, log_table_id, study_table_id):
    """Creates a database and tables for sample data."""
    spanner_client = spanner.Client(project="cameltrain")
    
    instance = spanner_client.instance(instance_id)
    if(instance.exists()):
        print("Instance with ID {} exists.".format(instance_id))
    else:
        config_name = "{}/instanceConfigs/regional-us-central1".format(
            spanner_client.project_name
        )

        instance = spanner_client.instance(
            instance_id,
            configuration_name=config_name,
            display_name="Log Data",
            node_count=1
        )

        operation = instance.create()

        print("Waiting for operation to complete...")
        operation.result(OPERATION_TIMEOUT_SECONDS)

        print("Created instance {}".format(instance_id))

    database = instance.database(database_id)    
    if(database.exists()):
        print("Database with ID {} exists.".format(database_id))
    else:
        operation = database.create()
        print("Waiting for operation to complete...")
        operation.result(OPERATION_TIMEOUT_SECONDS)
        print("Created database {} on instance {}".format(database_id, instance_id))
    
        #TODO : meetashah : seperate this table creation logic when update the google-cloud-spanner library
        #                   like one in spanner_test.py

        operation = database.update_ddl(
            [
                """CREATE TABLE """+log_table_id+""" (
                    Id     INT64 NOT NULL,
                    LogFormat    INT64,
                    LogPathPrefix     STRING(MAX),
                    LogOwner   STRING(MAX),
                    LogLabel   STRING(MAX)
                ) PRIMARY KEY (Id)"""
            ]
        )
        print("Waiting for operation to complete...")
        operation.result(OPERATION_TIMEOUT_SECONDS)
        print(
            "Created {} table on database {} on instance {}".format(
                log_table_id, database_id, instance_id
            )
        )

        operation = database.update_ddl(
            [
                """CREATE TABLE """+study_table_id+""" (
                    LogId     INT64 NOT NULL,
                    StudyName   STRING(MAX)
                ) PRIMARY KEY (LogId)"""
            ]
        )
        print("Waiting for operation to complete...")
        operation.result(OPERATION_TIMEOUT_SECONDS)
        print(
            "Created {} table on database {} on instance {}".format(
                study_table_id, database_id, instance_id
            )
        )

def Insert_In_StudyDetails_Table(study_details, instance_id, database_id, study_table_id):
    """adds study details to table mapped to unique LogId."""
    spanner_client = spanner.Client()
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)
    
    def insert_StudyDetails(transaction):
                
        query = f"INSERT {study_table_id} (LogId, StudyName) VALUES ({study_details['LogId']}, '{study_details['StudyName']}')"
        # print("StudyDetail query : ", query)

        row_ct = transaction.execute_update(query)
        print("{} record inserted to spanner table {}".format(row_ct, study_table_id))

    database.run_in_transaction(insert_StudyDetails)

def Fetch_From_StudyDetails_Table(log_id, instance_id, database_id, study_table_id):
    """fetch study name from table mapped to unique LogId."""
    spanner_client = spanner.Client()
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    with database.snapshot() as snapshot:

        query = f"SELECT StudyName FROM {study_table_id} WHERE LogId = {log_id}"
        results = snapshot.execute_sql(query)

        # print(results)
        for row in results:
            # print("For LogId : {} => StudyName: {}".format(log_id ,row[0]))
            return row[0]
        

def Insert_In_LogDetails_Table(new_log_entry, instance_id, database_id, table_id):
    """Writes in the sight service database to spanner table.

    Returns:
      
    """
    spanner_client = spanner.Client()
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    
    def insert_LogDetails(transaction):
        
        query = f"INSERT {table_id} (Id, LogFormat, LogPathPrefix, LogOwner, LogLabel) VALUES ({new_log_entry['Id']}, {new_log_entry['LogFormat']}, '{new_log_entry['LogPathPrefix']}', '{new_log_entry['LogOwner']}', '{new_log_entry['LogLabel']}')"
        # print("LogDetail query : ", query)

        row_ct = transaction.execute_update(query)
        print("{} record inserted to spanner table {}".format(row_ct, table_id))

    database.run_in_transaction(insert_LogDetails)

