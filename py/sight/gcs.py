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

from google.cloud import bigquery
from google.cloud import storage
from absl import logging

def uploadFileToGcsBucket(project_id, bucket_name, gcp_path, local_path, file_name, file_format):
    """ Upload file to gcs bucket.

    Args:
        project_id: GCP projectId.
        bucket_name: Bucket name in given project Id.
        gcp_path: Gcp diretory path in bucket where file will be stored in bucket.
        local_path: local diretory path of system where file is currently stored locally.
        file_name: Name of the file to upload.
        file_format: Format of the file to upload.

    Returns:
        
    """
    file = file_name+file_format
    # file = file_name
    logging.info('Uploading {} to bucket : {}.'.format(file, bucket_name))

    gcp_file_path = gcp_path+file
    local_file_path = local_path+file

    storage_client = storage.Client(project_id)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(gcp_file_path)
    blob.upload_from_filename(local_file_path)
    logging.info('{} file successfully uploaded to : {}.'.format(file, bucket_name))


def createExternalBQtable(project_id, dataset_name, table_name, external_file_format, external_file_uri):
    """ Create BigQuery external table mapping to file in GCS bucket.

    Args:
        project_id: GCP projectId.
        dataset_name: Dataset name in BigQuery.
        table_name: Table name of the table to be created.
        external_file_format: File format of external file from GCS bucket, whichl will be mapped to the external table.
        external_file_uri: File uri of external file from GCS bucket, whichl will be mapped to the external table.

    Returns:
        
    """
    logging.info('Creating external table {} mapping to : {}.'.format(table_name, external_file_uri))
    client = bigquery.Client(project_id)
    dataset_ref = client.dataset(dataset_name)
    table_ref = bigquery.TableReference(dataset_ref, table_name)
    table = bigquery.Table(table_ref)

    external_config = bigquery.ExternalConfig(external_file_format)
    external_config.source_uris = [external_file_uri] 
    table.external_data_configuration = external_config
    client.create_table(table)
    logging.info('{} table successfully created.'.format(table_name))
