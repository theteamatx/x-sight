"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_darts_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of
train_darts_surrogate.
"""

import os
from typing import Optional, Dict, Sequence, Tuple

from absl import app
from absl import flags
import glob
from google.cloud import bigquery
import pandas as pd
import os.path
import subprocess

_LOG_ID = flags.DEFINE_string(
    'log_id',
    '',
    'Unique ID of the simulation run log.',
)
_OUTFILE = flags.DEFINE_string(
    'outfile',
    '',
    'Path of the file where the run simulation time series will be stored.',
)
_PROJECT_ID = flags.DEFINE_string(
    'project_id', os.environ['PROJECT_ID'], 'ID of the current GCP project.'
)


def build_query(raw_query, params: dict = None):
  """Format query using given parameters.

  If no parameters are provided the query is returned as is.

  Args:
    raw_query: raw sql query
    params: optional query parameters

  Returns:
    query with parameters inserted
  """
  query = raw_query
  if params is not None:
    query = query.format(**params)
  print(query)
  return query

def run_query(query_file_name: str, bq_client: bigquery.Client, params: Dict=dict()) -> pd.DataFrame:
  current_script_directory = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(
      current_script_directory, query_file_name+'.sql'
  )) as file:
    query_template = file.read()

  return bq_client.query(
      build_query(
          query_template,
          {'log_id': _LOG_ID.value} | params,
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

      
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  bq_client = bigquery.Client(project=_PROJECT_ID.value)

  if os.path.isfile(_OUTFILE.value+'.states'):
    states = pd.read_csv(_OUTFILE.value+'.states')
    variables = pd.read_csv(_OUTFILE.value+'.variables')
  else:
    for type in ['autoreg', 'boundary', 'initial']:
      run_query(f'sim_named_{type}_var', bq_client)
      run_query('sim_value', bq_client, {'type': type})
      run_query('sim_all_vars', bq_client, {'type': type})
    
    autoreg_variables = load_table(f'cameltrain.sight_logs.{_LOG_ID.value}_autoreg_all_vars_log', bq_client)['label'].tolist()
    boundary_variables = load_table(f'cameltrain.sight_logs.{_LOG_ID.value}_boundary_all_vars_log', bq_client)['label'].tolist()
    initial_variables = load_table(f'cameltrain.sight_logs.{_LOG_ID.value}_initial_all_vars_log', bq_client)['label'].tolist()
    print('autoreg_variables(#%d)=%s' % (len(autoreg_variables), autoreg_variables))
    print('boundary_variables(#%d)=%s' % (len(boundary_variables), boundary_variables))
    print('initial_variables(#%d)=%s' % (len(initial_variables), initial_variables))


    for type in ['autoreg', 'boundary', 'initial']:
      run_query('sim_unordered_time_series', bq_client, {'type': type})

    run_query('sim_ordered_time_series', bq_client, {'num_autoreg_vars': len(autoreg_variables),
                                                     'num_boundary_vars': len(boundary_variables),
                                                     'num_initial_vars': len(initial_variables),
                                                     })
    
    extract_job = bq_client.extract_table(
        bigquery.DatasetReference(_PROJECT_ID.value, 'sight_logs').table(f'{_LOG_ID.value}_simulation_ordered_time_series_log'),
        f'gs://{_PROJECT_ID.value}-sight/sight-logs/{_LOG_ID.value}.sim_ordered_time_series.*.csv',
        # Location must match that of the source table.
        location="US",
    )  # API request
    extract_job.result()  # Waits for job to complete.

    out = subprocess.run(
        ['gsutil', 'cp', 
         f'gs://{_PROJECT_ID.value}-sight/sight-logs/*{_LOG_ID.value}.sim_ordered_time_series.*.csv',
         '/tmp'],
        capture_output=True,
        # check=True,
    )
    print(out)
    
    time_series = []
    for i, ts_file in enumerate(glob.glob(f'/tmp/*{_LOG_ID.value}.sim_ordered_time_series.*.csv')):
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
      cur_ts.to_csv(_OUTFILE.value+'.'+str(i), index=False)
      print('cur_ts=', cur_ts)
      time_series.append(cur_ts)
    ts_df = pd.concat(time_series, axis=0)
    # ts_df[variables] = ts_df['values'].str.split(',', expand=True)
    
    print(ts_df)
    # return

    # states = bq_client.query(
    #     build_query(
    #         'SELECT *  FROM `cameltrain.sight_logs.{log_id}_simulation_ordered_time_series_log`',
    #         {'log_id': _LOG_ID.value},
    #     )).to_dataframe()
    
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', 1)
    # print('states=%s' % states)
    if _OUTFILE.value:
      ts_df.reset_index().to_csv(_OUTFILE.value, index=False)
      # variables.to_csv(_OUTFILE.value+'.variables', index=False)
  
  # value_types = pd.DataFrame(states['value_types'].tolist(), columns=variables).reset_index()
  # double_values = pd.DataFrame(states['double_values'].tolist(), columns=variables).reset_index()
  # string_values = pd.DataFrame(states['string_values'].tolist(), columns=variables).reset_index()
  # for v in variables:
  #   print('### %s: %s ' % (v, value_types[v][0]))
  #   if value_types[v][0] == 'ST_STRING':
  #     possible_values = string_values[v].unique()
  #     print('possible_values=', possible_values)
  #     df_encoded = pd.get_dummies(string_values[[v]], columns=[v, ])
  #     print('df_encoded=', df_encoded)

  # states = states.drop(columns=['labels', 'double_values', 'string_values', 'value_types'])
  # all_dfs = [states[['time_step_index']]]
  # for v in variables:
  #   if value_types[v][0] == 'ST_STRING':
  #     # states[v] = string_values[v]
  #     all_dfs.append(pd.get_dummies(string_values[[v]], columns=[v, ]))
  #     # print('df_encoded=', df_encoded)
  #   elif value_types[v][0] == 'ST_DOUBLE':
  #     # states[v] = double_values[v]
  #     all_dfs.append(double_values[[v]])
  # states = pd.concat(all_dfs, axis=1)
  # print('states=', states)

  # if _OUTFILE.value:
  #   states.to_csv(_OUTFILE.value, index=False)

if __name__ == '__main__':
  app.run(main)
