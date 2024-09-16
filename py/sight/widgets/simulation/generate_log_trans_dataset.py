from typing import Sequence, Tuple

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
import csv
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import random as rn

_INPUT_PATH = flags.DEFINE_string(
    'input_path',
    '',
    'The path where the input experiments csv file is located.',
)
_OUTPUT_PATH_TRAIN = flags.DEFINE_string(
    'output_train_path',
    '',
    'The path where the output training data for the transformer model file will be written.',
)
_OUTPUT_PATH_TRAIN_DATA = flags.DEFINE_string(
    'output_train_data_path',
    '',
    'The path where the output raw training data rows file will be written.',
)
_OUTPUT_PATH_VAL = flags.DEFINE_string(
    'output_val_path',
    '',
    'The path where the output validation data for the transformer model file will be written.',
)
_OUTPUT_PATH_VAL_DATA = flags.DEFINE_string(
    'output_val_data_path',
    '',
    'The path where the output raw validation data rows file will be written.',
)
_OUTPUT_META_PATH = flags.DEFINE_string(
    'output_meta_path',
    '',
    'The path where the output metadata data file will be written.',
)
_HIST_LEN = flags.DEFINE_integer(
    'hist_len',
    5,
    'The path where the output validation data file will be written.',
)
_TRAIN_FRAC = flags.DEFINE_float(
    'train_frac',
    .8,
    'The path where the output validation data file will be written.',
)


@dataclass
class Dataset:
    train_df: pd.DataFrame
    train_data_df: pd.DataFrame
    validate_df: pd.DataFrame
    validate_data_df: pd.DataFrame
    max_input_len: int
    max_pred_len: int


def generate_prediction(row: np.ndarray, columns: Sequence[str]) -> str:
    """Returns the representation of row to use as the string the transformer will predict.

  Arguments:
    row: The data row, containing data for all the columns.
    columns: The names of the columns, one for each row element. Their names include the
      prefix 'autoreg:', 'boundary:' or 'initial:' to indicate their role in the simulation.
  """
    data = []
    for i, c in enumerate(columns):
        if c.startswith('autoreg:'):
            data.append(str(row[i]))
    return ' '.join(data)


def generate_input(rows: Sequence[np.ndarray], next_row: np.ndarray,
                   columns: Sequence[str]) -> str:
    """Returns the representation of rows to use as the string the transformer will take as input.

  Arguments:
    rows: The data rows, containing data for all columns.
    columns: The names of the columns, one for each row element. Their names include the
      prefix 'autoreg:', 'boundary:' or 'initial:' to indicate their role in the simulation.
  """
    # print('rows: ', type(rows))
    # print('columns: ', type(columns))
    out = ''
    for row_idx, row in enumerate(rows):
        if row_idx == 0:
            out += 'initial:'
            # print('row=', row)
            # print('columns=', columns)
            for i, c in enumerate(columns):
                # print(i,': ', i, ' row[i]=', row[i], ' c=', c)
                if c.startswith('initial:'):
                    out += ' ' + str(row[i])
            out += ', '
        else:
            out += '| '
        out += 'boundary:'
        for i, c in enumerate(columns):
            if c.startswith('boundary:'):
                out += ' ' + str(row[i])
        out += ', autoreg:'
        for i, c in enumerate(columns):
            if c.startswith('autoreg:'):
                out += ' ' + str(row[i])
    out += '| boundary:'
    for i, c in enumerate(columns):
        if c.startswith('boundary:'):
            out += ' ' + str(next_row[i])

    return out


def build_dataset(sim_log: pd.DataFrame, hist_len: int,
                  train_frac: float) -> Dataset:
    """Loads the simulation log dataset and splits it into a training and a validation set.

  Arguments:
    sim_log: The full log that contains the time series of all simiulation runs.
    hist_len: the number of time steps to use as input for each model
      prediction.
    train_frac: the fraction of the dataset to use for training.

  Returns:
    The training and validation datasets, each of which has columns input and target.
  """
    simulations = sim_log.groupby(['sim_location'])

    train_inputs = []
    train_preds = []
    train_data = []
    validate_inputs = []
    validate_preds = []
    validate_data = []
    max_input_len = 0
    max_pred_len = 0

    for _, sim_log in simulations:
        if rn.random() < train_frac:
            inputs = train_inputs
            preds = train_preds
            data = train_data
        else:
            inputs = validate_inputs
            preds = validate_preds
            data = validate_data

        hist = []
        data_columns = list(sim_log.columns[3:])
        for idx in range(sim_log.shape[0]):
            cur_row = sim_log.iloc[idx].values.astype(str)
            data.append(cur_row)
            # logging.info('inputs(#%d)=%s', len(cur_row), cur_row)
            if len(hist) == hist_len:
                # next_input = ' '.join(hist)
                input = generate_input(hist, cur_row[3:], data_columns)
                prediction = generate_prediction(cur_row[3:], data_columns)

                max_input_len = max(max_input_len, len(input))
                inputs.append(input)

                max_pred_len = max(max_pred_len, len(prediction))
                preds.append(prediction)

                hist.pop(0)
            hist.append(cur_row[3:])
        # logging.info('inputs(#%d)=%s', len(inputs), inputs)
        # logging.info('preds(#%d)=%s', len(preds), preds)

    train_df = pd.DataFrame({
        'input': train_inputs,
        'pred': train_preds,
    })

    validate_df = pd.DataFrame({
        'input': validate_inputs,
        'pred': validate_preds,
    })

    return Dataset(
        train_df,
        pd.DataFrame(train_data),
        validate_df,
        pd.DataFrame(validate_data),
        max_input_len,
        max_pred_len,
    )


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    dataset = build_dataset(
        sim_log=pd.read_csv(_INPUT_PATH.value),
        hist_len=_HIST_LEN.value,
        train_frac=_TRAIN_FRAC.value,
    )
    dataset.train_df.to_csv(_OUTPUT_PATH_TRAIN.value,
                            index=False,
                            quoting=csv.QUOTE_ALL)
    dataset.train_data_df.to_csv(_OUTPUT_PATH_TRAIN_DATA.value,
                                 index=False,
                                 quoting=csv.QUOTE_ALL)
    dataset.validate_df.to_csv(_OUTPUT_PATH_VAL.value,
                               index=False,
                               quoting=csv.QUOTE_ALL)
    dataset.train_data_df.to_csv(_OUTPUT_PATH_VAL_DATA.value,
                                 index=False,
                                 quoting=csv.QUOTE_ALL)

    with open(_OUTPUT_META_PATH.value, 'w') as f:
        json.dump(
            {
                'max_input_len': dataset.max_input_len,
                'max_pred_len': dataset.max_pred_len,
            }, f)


if __name__ == '__main__':
    app.run(main)
