"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_darts_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of
train_darts_surrogate.
"""

import os
from typing import Optional, Sequence, Tuple

from absl import app
from absl import flags
from google.cloud import bigquery
import pandas as pd

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

def run_query(query_file_name: str, bq_client: bigquery.Client) -> pd.DataFrame:
  current_script_directory = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(
      current_script_directory, query_file_name+'.sql'
  )) as file:
    query_template = file.read()

  return bq_client.query(
      build_query(
          query_template,
          {'log_id': _LOG_ID.value},
      )).to_dataframe()
      
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  bq_client = bigquery.Client(project=_PROJECT_ID.value)

  current_script_directory = os.path.dirname(os.path.abspath(__file__))
  # run_query('sim_named_var', bq_client)
  # run_query('sim_value', bq_client)
  # run_query('sim_all_vars', bq_client)
  # run_query('sim_unordered_time_series', bq_client)
  # run_query('sim_ordered_time_series', bq_client)

  variables = bq_client.query(
      build_query( 
          'SELECT * FROM `cameltrain.sight_logs.{log_id}_all_vars_log`',
          {'log_id': _LOG_ID.value},
      )).to_dataframe()['label'].tolist()
  # variables = ['AFTER_TREATMENT_BA', 'AFTER_TREATMENT_CCF', 'AFTER_TREATMENT_RES_QMD', 'AFTER_TREATMENT_SDI', 'AFTER_TREATMENT_TOP_HT', 'FERTILIZ', 'GROWTH_THIS_PERIOD_ACCRE_PER', 'GROWTH_THIS_PERIOD_MORT_YEAR', 'GROWTH_THIS_PERIOD_PERIOD_YEARS', 'REMOVALS_MERCH_BD_FT', 'REMOVALS_MERCH_CU_FT', 'REMOVALS_NO_OF_TREES', 'REMOVALS_TOTAL_CU_FT', 'STAND_ATTRIBUTES_CROWN_COMP_FACTOR', 'STAND_ATTRIBUTES_NASAL_AREA(M2/HA)', 'STAND_ATTRIBUTES_QUADRATIC_MEAN_DBH(CM)', 'STAND_ATTRIBUTES_STAND_AGE', 'STAND_ATTRIBUTES_TOPHT_LARGEST_40/HA', 'STAND_ATTRIBUTES_TREES_PER_HA', 'START_OF_SIMULATION_PERIOD_BA', 'START_OF_SIMULATION_PERIOD_CCF', 'START_OF_SIMULATION_PERIOD_MERCH_BD_FT', 'START_OF_SIMULATION_PERIOD_MERCH_CU_FT', 'START_OF_SIMULATION_PERIOD_NO_OF_TREES', 'START_OF_SIMULATION_PERIOD_QMD', 'START_OF_SIMULATION_PERIOD_SDI', 'START_OF_SIMULATION_PERIOD_TOP_HT', 'START_OF_SIMULATION_PERIOD_TOTAL_CU_FT', 'Tree Mean', 'Tree Num', 'num_trees_planted', 'plant_seedling_age', 'plant_seedling_height', 'plant_survival', 'thin_dbh_max', 'thin_dbh_min', 'thin_efficiency']
  print('variables=', variables)

  states = bq_client.query(
      build_query(
          'SELECT *  FROM `cameltrain.sight_logs.{log_id}_simulation_unordered_time_series_log` WHERE ARRAY_LENGTH(value_types)='+str(len(variables))+' ORDER BY sim_location, time_step_index',
          {'log_id': _LOG_ID.value},
      )).to_dataframe()
  
  pd.set_option('display.max_columns', None)
  # pd.set_option('display.max_rows', 1)
  # print('states=%s' % states)
  if _OUTFILE.value:
    states.to_csv(_OUTFILE.value+'.states', index=False)
  
  value_types = pd.DataFrame(states['value_types'].tolist(), columns=variables).reset_index()
  double_values = pd.DataFrame(states['double_values'].tolist(), columns=variables).reset_index()
  string_values = pd.DataFrame(states['string_values'].tolist(), columns=variables).reset_index()
  for v in variables:
    print('### %s: %s ' % (v, value_types[v][0]))
    if value_types[v][0] == 'ST_STRING':
      possible_values = string_values[v].unique()
      print('possible_values=', possible_values)
      df_encoded = pd.get_dummies(string_values[[v]], columns=[v, ])
      print('df_encoded=', df_encoded)



  states = states.drop(columns=['labels', 'double_values', 'string_values', 'value_types'])
  all_dfs = [states[['time_step_index']]]
  for v in variables:
    if value_types[v][0] == 'ST_STRING':
      # states[v] = string_values[v]
      all_dfs.append(pd.get_dummies(string_values[[v]], columns=[v, ]))
      # print('df_encoded=', df_encoded)
    elif value_types[v][0] == 'ST_DOUBLE':
      # states[v] = double_values[v]
      all_dfs.append(double_values[[v]])
  states = pd.concat(all_dfs, axis=1)
  print('states=', states)

  if _OUTFILE.value:
    states.to_csv(_OUTFILE.value, index=False)

if __name__ == '__main__':
  app.run(main)
