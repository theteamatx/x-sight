"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_darts_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of
train_darts_surrogate.
"""

from typing import List, Sequence, Tuple

from absl import app
from absl import flags
from darts import TimeSeries
from darts.metrics.metrics import mae
from darts.models import LightGBMModel
from darts.models import LinearRegressionModel
from darts.models import NBEATSModel
from darts.models import RandomForest
from darts.models import RNNModel
from darts.models.forecasting.block_rnn_model import BlockRNNModel
import numpy as np
import pandas as pd

_INFILE = flags.DEFINE_string(
    'infile',
    '',
    'Path of the file where the run simulation time series will be loaded'
    ' from.',
)
_BOUNDARY_COND_VARS = flags.DEFINE_list(
    'boundary_cond_vars',
    [],
    'The variables that document the run boundary conditions.',
)
_OUT_MODEL_FILE = flags.DEFINE_string(
    'out_model_file',
    '',
    'Path of the file where the run simulation time series will be loaded'
    ' from.',
)
#'GF': '1601933670369823365'


def create_list_of_timeseries(df: pd.DataFrame) -> List[pd.DataFrame]:
  series = []
  for idx, sim_data in df.groupby('sim_location'):
    series.append(sim_data)
    # if num_vars is None:
    #   num_vars = len(sim_data['values'][0])
  return series


def create_darts_time_series(
    series: List[pd.DataFrame],
    state_vars: List[str],
    boundary_cond_vars: List[str],
) -> List[TimeSeries]:
  time_series = []
  for si, s in enumerate(series):
    s_data = []
    if boundary_cond_vars:
      for i, s_row in s.iterrows():
        row = []
        for v in boundary_cond_vars:
          row.append(s_row[v])
        for v in state_vars:
          row.append(0)
        s_data.append(row)

    for i, s_row in s.iterrows():
      row = []
      if boundary_cond_vars:
        for v in boundary_cond_vars:
          row.append(s_row[v])
      for v in state_vars:
        row.append(s_row[v])
      s_data.append(row)

    lagged_s = pd.DataFrame(s_data, columns=boundary_cond_vars +
                            state_vars).reset_index()
    # print('lagged_s', lagged_s)
    time_series.append(
        TimeSeries.from_dataframe(lagged_s, 'index',
                                  boundary_cond_vars + state_vars))

  return time_series


def train_model(time_series: List[TimeSeries]):
  # model = LinearRegressionModel(lags=[-40, -4, -3, -2, -1],
  #                               output_chunk_length=1,
  #                               multi_models=True)
  model = RandomForest(
      lags=[-6, -5, -4, -3, -2, -1],
      output_chunk_length=10,
      multi_models=True,
  )

  # model = LightGBMModel(lags=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1],
  #                    output_chunk_length=10,
  #                    multi_models=True)
  # model = RNNModel(
  #     model="LSTM",
  #     input_chunk_length=10,
  #     training_length=10,
  #     n_epochs=500,
  # )
  # model = BlockRNNModel(
  #     input_chunk_length=41,
  #     output_chunk_length=41,
  #     n_rnn_layers=2,
  #     n_epochs=300,
  # )
  # print(time_series)
  model.fit(time_series)
  print('model=', model)
  return model


def eval_model(model, time_series, split_point) -> float:
  avg_err = 0
  for pred_idx in range(len(time_series)):
    total_steps = len(time_series[pred_idx])
    # print(time_series[pred_idx])
    train, val = time_series[pred_idx].split_before(split_point)
    prediction = model.predict(total_steps - split_point,
                               series=train,
                               num_samples=1)

    # print (prediction)
    avg_err += mae(prediction, val) / len(time_series)
    print('   %s : %s' % (pred_idx, mae(prediction, val)))

    # import matplotlib.pyplot as plt
    # # for v in ts_vars:
    # time_series[pred_idx][v].plot(label='actual '+v)
    # prediction[v].plot(label='forecast '+v, lw=3)
    # # plt.legend()

    # plt.show()
    # break
  print('%s' % (avg_err,))
  return avg_err


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_df = pd.read_csv(_INFILE.value)

  control_vars = ['sim_location', 'time_step_index', 'next_time_step_index']
  state_vars = [
      v for v in dataset_df.columns
      if v not in _BOUNDARY_COND_VARS.value and v not in control_vars
  ]
  # pd.set_option('display.max_columns', None)
  # print(dataset_df)
  # return

  time_series = create_darts_time_series(create_list_of_timeseries(dataset_df),
                                         state_vars, _BOUNDARY_COND_VARS.value)

  model = train_model(time_series)

  error = eval_model(model, time_series, 10)


if __name__ == '__main__':
  app.run(main)
