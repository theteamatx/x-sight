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

"""GCS related helper functions."""

import os
import subprocess

from absl import logging
from google.cloud import bigquery
from google.cloud import storage
from sight.proto import sight_pb2


def upload_blob_from_stream(bucket_name, gcp_path, file_obj, file_name, count):
  """uploads given file to the bucket.

  Args:
    bucket_name: name of the bucket to store the file
    gcp_path: directory path to store the file
    file_obj: file object to be stored
    file_name: name given to file
    count: chunk number of file
  """
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  if not bucket.exists():
    # logging.info(f"creating bucket {bucket_name}, as it didn't exist....")
    bucket = storage_client.create_bucket(bucket_name)

  blob_name = gcp_path + '/' + file_name + '_' + str(count) + '.avro'
  blob = bucket.blob(blob_name)
  # Rewind the stream to the beginning. This step can be omitted if the input
  # stream will always be at a correct position.
  file_obj.seek(0)
  # Upload data from the stream to your bucket.
  blob.upload_from_file(file_obj)
  # logging.info(f'Stream data uploaded to {blob_name} in bucket {bucket_name}.')


def create_table(
    project_id,
    dataset_name,
    table_name,
    external_file_format,
    external_file_uri,
):
  """Create BigQuery external table mapping to file in GCS bucket.

  Args:
      project_id: GCP projectId.
      dataset_name: Dataset name in BigQuery.
      table_name: Table name of the table to be created.
      external_file_format: File format of external file from GCS bucket, which
        will be mapped to the external table.
      external_file_uri: File uri of external file from GCS bucket, whichl will
        be mapped to the external table.

  Returns:
  """

  try:
    # Check if the dataset exists
    client = bigquery.Client(project_id)
    dataset = client.get_dataset(dataset_name)
    # logging.info(f"Dataset {dataset_name} already exists.")
  except Exception as e:
    # If the dataset does not exist, create a new dataset
    dataset = bigquery.Dataset(f"{project_id}.{dataset_name}")
    dataset = client.create_dataset(dataset)
  #   logging.info(f"Dataset {dataset_name} created.")


  # logging.info(
  #     'Creating external table %s mapping to : %s.',
  #     table_name,
  #     external_file_uri,
  # )
  try:
    client = bigquery.Client(project_id)
    dataset_ref = client.dataset(dataset_name)
    table_ref = bigquery.TableReference(dataset_ref, table_name)
    table = bigquery.Table(table_ref)

    external_config = bigquery.ExternalConfig(external_file_format)
    external_config.source_uris = [external_file_uri]
    table.external_data_configuration = external_config
    client.create_table(table)
    # logging.info('%s table successfully created.', table_name)
  except:
    logging.info(f"Error creating table: {e}")


def create_external_bq_table(
    params: sight_pb2.Params, file_name: str, client_id: int
):
  """create external table in BigQuery from avro files using URI, located in the bucket.

  Args:
    params: sight parameters to get details of the files
    file_name: name of the file
    client_id: sight client id
  """
  external_file_uri = (
      params.external_file_uri
      + params.bucket_name
      + '/'
      + params.gcp_path
      + '/'
      # + '/client_'
      + params.label
      + '_'
      + str(client_id)
      + '/'
      + '*'
      + params.file_format
  )
  if 'PARENT_LOG_ID' not in os.environ:
    create_table(
        os.environ["PROJECT_ID"],
        params.dataset_name,
        file_name,
        params.external_file_format,
        external_file_uri,
    )
