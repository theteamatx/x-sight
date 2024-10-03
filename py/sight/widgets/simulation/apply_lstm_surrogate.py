"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_lstm_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of train_lstm_surrogate.
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from google.cloud import bigquery
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import tensorflow as tf

_PROJECT_ID = flags.DEFINE_string('project_id', os.environ['PROJECT_ID'],
                                  "ID of the current GCP project.")
_LOG_ID = flags.DEFINE_string('log_id', '', "ID of the log being analyzed.")
_MODEL_IN_PATH = flags.DEFINE_string('model_in_path', '',
                                     'Path where the trained model is stored.')
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 50, 'Number of steps of history in the prediction.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch size.')


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
  _SCHEMA_FILE_PATH = os.path.join(current_script_directory,
                                   'simulation_time_series.sql')

  with open(f'x-sight/py/sight/widgets/simulation/simulation_time_series.sql'
           ) as file:
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

  next_state = []
  for i, d in enumerate(sim_dataset):
    next_state.append(d[1])
  next_state = tf.concat(next_state, 0)
  # next_state = np.array(next_state)
  # print('next_state(#%d)=%s' % (len(next_state), next_state))

  model = keras.models.load_model(_MODEL_IN_PATH.value)
  prediction = np.array(model.predict(sim_dataset))
  print('prediction=%s=%s' % (prediction.shape, prediction))

  error_sum = 0
  num_pred = 0
  for i, p in enumerate(prediction):
    # print ('%s: next_state=%s prediction=%s' %
    #        (
    #            np.linalg.norm(next_state[i] - p, ord=2),
    #            next_state[i], p
    #            ))
    error_sum += np.linalg.norm(next_state[i] - p, ord=2)
    num_pred += 1
  print('error = %s' % (error_sum / num_pred))

  print('error = %s' % np.linalg.norm(next_state - prediction, ord=2))

  # model.save(_MODEL_OUT_PATH.value)


if __name__ == "__main__":
  app.run(main)
