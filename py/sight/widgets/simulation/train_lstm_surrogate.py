"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_lstm_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of train_lstm_surrogate.
"""

from typing import Sequence

from absl import app
from absl import flags
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import os

from google.cloud import bigquery

_PROJECT_ID = flags.DEFINE_string(
    'project_id', os.environ['PROJECT_ID'], "ID of the current GCP project."
)
_LOG_ID = flags.DEFINE_string(
    'log_id', '', "ID of the log being analyzed."
)
_MODEL_OUT_PATH = flags.DEFINE_string(
    'model_out_path', '', 'Path where the trained model should be stored.'
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 50, 'Number of steps of history in the prediction.'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 128, 'Batch size.'
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
    return query

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  current_script_directory = os.path.dirname(os.path.abspath(__file__))
  _SCHEMA_FILE_PATH = os.path.join(current_script_directory, 'simulation_time_series.sql')

  with open(f'x-sight/py/sight/widgets/simulation/simulation_time_series.sql') as file:
    query_template = file.read()
  
  query = build_query(query_template, {'log_id': _LOG_ID.value})
  print('query=%s' % query)
  bq_client = bigquery.Client(project=_PROJECT_ID.value)
  df = bq_client.query(query).to_dataframe()
  
  sim_dataset = None
  simulations = df.groupby(by=['sim_location'])
  for sim_location, sim_data in simulations:
    # print('sim_location=%s, sim_data=%s' % (sim_location, sim_data))
    df = pd.DataFrame(sim_data['values'].to_list())
    input_data = df[:-_NUM_STEPS.value]
    targets = df[_NUM_STEPS.value:]
    
    cur_dataset = keras.utils.timeseries_dataset_from_array(
        input_data, 
        targets, 
        sequence_length=_NUM_STEPS.value, 
        batch_size=_BATCH_SIZE.value)
    
    if sim_dataset is None:
      sim_dataset = cur_dataset
    else:
      sim_dataset = sim_dataset.concatenate(cur_dataset)
  
  # sim_dataset = tf.concat(sim_datasets, axis=0)
  print('sim_dataset=%s' % sim_dataset)
  # return 
  # data = pd.DataFrame(df['values'].to_list())
  # l = data.values.tolist()
  # for i in range(16):
  #   print ('%s: %s' % (i, l[i]))

  model = Sequential()
  model.add(LSTM(100, activation='relu', input_shape=(_NUM_STEPS.value, 20)))
  # model.add(LSTM(100, activation='relu', input_shape=(None, 100)))
  model.add(Dense(20))
  model.compile(optimizer='adam', loss='mse')
  
  h = model.fit(sim_dataset, steps_per_epoch=1, epochs=600, verbose=0)
  print('hist=%s' % h.history)
  
  # prediction = model.predict(dataset)
  # print('prediction=%s' % prediction)
  
  model.save(_MODEL_OUT_PATH.value)


if __name__ == "__main__":
  app.run(main)
