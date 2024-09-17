"""Queries the Simulation entries in a Sight log and saves them as a time series."""

import os
from typing import Optional, Dict, Sequence, Tuple

from absl import app
from dataclasses import dataclass
import glob
from google.cloud import bigquery
import pandas as pd
import os.path
import random as rn
import subprocess

def build_query(raw_query, params: dict = None):
  """Format query using given parameters.

  If no parameters are provided the query is returned as is.

  Args:
    raw_query: raw sql query.
    params: optional query parameters.

  Returns:
    Query with parameters inserted.
  """
  query = raw_query
  if params is not None:
    query = query.format(**params)
  print(query)
  return query

def run_query(query_file_name: str, bq_client: bigquery.Client, params: Dict=dict()) -> pd.DataFrame:
  """Runs the BigQuery in a file.
  
  Arguments:
    query_file_name: Name of the file within the current directory.
    bg_client: The client for accessing BigQuery.
    params: The fillable-parameters of the query.
  
  Returns:
    Dataframe with query results.
  """
  current_script_directory = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(
      current_script_directory, query_file_name+'.sql'
  )) as file:
    query_template = file.read()

  return bq_client.query(
      build_query(
          query_template,
          params,
      )).to_dataframe()

def load_table(table_name: str, bq_client: bigquery.Client) -> pd.DataFrame:
  """Queries a table and returns a DataFrame with its contents.

  Arguments:
    table_name: The fully-qualified name of the table to be read.
    bq_client: Object via which the BigQuery API will be accessed.
  """
  return bq_client.query(
      build_query(
          f'SELECT * FROM `{table_name}`',
      )).to_dataframe()     

      
def load_ts(log_id: str, project_id: str) -> pd.DataFrame:
  bq_client = bigquery.Client(project=project_id)

  for type in ['autoreg', 'boundary', 'initial']:
    run_query(f'sim_named_{type}_var', bq_client, {'log_id': log_id})
    run_query('sim_value', bq_client, {'type': type, 'log_id': log_id})
    run_query('sim_all_vars', bq_client, {'type': type, 'log_id': log_id})
  
  autoreg_variables = load_table(f'cameltrain.sight_logs.{log_id}_autoreg_all_vars_log', bq_client)['label'].tolist()
  boundary_variables = load_table(f'cameltrain.sight_logs.{log_id}_boundary_all_vars_log', bq_client)['label'].tolist()
  initial_variables = load_table(f'cameltrain.sight_logs.{log_id}_initial_all_vars_log', bq_client)['label'].tolist()
  print('autoreg_variables(#%d)=%s' % (len(autoreg_variables), autoreg_variables))
  print('boundary_variables(#%d)=%s' % (len(boundary_variables), boundary_variables))
  print('initial_variables(#%d)=%s' % (len(initial_variables), initial_variables))

  for type in ['autoreg', 'boundary', 'initial']:
    run_query('sim_unordered_time_series', bq_client, {'type': type, 'log_id': log_id})

  run_query('sim_ordered_time_series', bq_client, {'num_autoreg_vars': len(autoreg_variables),
                                                    'num_boundary_vars': len(boundary_variables),
                                                    'num_initial_vars': len(initial_variables),
                                                    'log_id': log_id
                                                    })
  
  extract_job = bq_client.extract_table(
      bigquery.DatasetReference(project_id, 'sight_logs').table(f'{log_id}_simulation_ordered_time_series_log'),
      f'gs://{project_id}-sight/sight-logs/{log_id}.sim_ordered_time_series.*.csv',
      # Location must match that of the source table.
      location="US",
  )  # API request
  extract_job.result()  # Waits for job to complete.

  out = subprocess.run(
      ['gsutil', 'cp', 
        f'gs://{project_id}-sight/sight-logs/*{log_id}.sim_ordered_time_series.*.csv',
        '/tmp'],
      capture_output=True,
      # check=True,
  )
  print(out)
  
  time_series = []
  for i, ts_file in enumerate(glob.glob(f'/tmp/*{log_id}.sim_ordered_time_series.*.csv')):
    print(ts_file)
    # print(pd.read_csv(ts_file))
    cur_ts = pd.read_csv(ts_file)
    # cur_ts.set_index(['sim_location', 'time_step_index'], inplace=True)
    # print('cur_ts.dtypes=', cur_ts.dtypes)
    cur_ts[[f'autoreg:{v}' for v in autoreg_variables]] = cur_ts['autoreg_values'].str.split(',', expand=True)
    cur_ts.drop(columns='autoreg_values', inplace=True)
    cur_ts[[f'boundary:{v}' for v in boundary_variables]] = cur_ts['boundary_values'].str.split(',', expand=True)
    cur_ts.drop(columns='boundary_values', inplace=True)
    cur_ts[[f'initial:{v}' for v in initial_variables]] = cur_ts['initial_values'].str.split(',', expand=True)
    cur_ts.drop(columns='initial_values', inplace=True)
            
    # tree_species_dummies = pd.get_dummies(cur_ts['Tree Species'], columns=['TreeSpecies', ])
    # cur_ts.drop(columns='Tree Species', inplace=True)
    # cur_ts = pd.concat([cur_ts, tree_species_dummies], axis=1)

    print('cur_ts.columns=%s=%s' %(len(cur_ts.columns), cur_ts.columns.tolist()))
    time_series.append(cur_ts)
  return pd.concat(time_series, axis=0)
  

@dataclass
class TsDataset:
  train: pd.DataFrame
  validate: pd.DataFrame

def split_time_series(ts: pd.DataFrame, train_frac: float) -> TsDataset:
  """Splits a simulation time series into train and validation subsets.
  
  The split occurs at the granularity of whole simulation runs.

  Arguments:
    ts: Dataframe with all time series observations.
    train_frac: The fraction of simulation runs assigned to the training set.
  
  Returns:
    The full dataset with separated validation and training subsets.
  """
  simulations = ts.groupby(['sim_location'])
  train = []
  validate = []
  for _, sim_log in simulations:
    if rn.random() < train_frac:
      train.append(sim_log)
    else:
      validate.append(sim_log)
  return TsDataset(pd.concat(train, axis=0) if train else pd.DataFrame(),
                   pd.concat(validate, axis=0) if validate else pd.DataFrame())