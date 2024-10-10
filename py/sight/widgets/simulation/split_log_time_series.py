"""TODO(bronevet): DO NOT SUBMIT without one-line documentation for train_darts_surrogate.

TODO(bronevet): DO NOT SUBMIT without a detailed description of
train_darts_surrogate.
"""

import glob
import os
import os.path
import subprocess
from typing import Dict, Optional, Sequence, Tuple

from absl import app
from absl import flags
from google.cloud import bigquery
import pandas as pd

_LOG_ID = flags.DEFINE_string(
    'log_id',
    '',
    'Unique ID of the simulation run log.',
)
_INFILE = flags.DEFINE_string(
    'infile',
    '',
    'Path of the file where the input run simulation time series is be stored.',
)

_TRAIN_OUTFILE = flags.DEFINE_string(
    'train_outfile',
    '',
    'Path of the file where the training run simulation time series will be stored.',
)
_VALIDATE_OUTFILE = flags.DEFINE_string(
    'validate_outfile',
    '',
    'Path of the file where the validation run simulation time series will be stored.',
)
_SPLIT_COL = flags.DEFINE_string(
    'split_col',
    '',
    'The column on which to split training and validation.',
)
_SPLIT_VALIDATE_VAL = flags.DEFINE_string(
    'split_validate_val',
    '',
    'The value for the split column that will be assigned to the validation dataset.',
)

_PROJECT_ID = flags.DEFINE_string('project_id', os.environ['PROJECT_ID'],
                                  'ID of the current GCP project.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_df = pd.read_csv(_INFILE.value)
  train_df = data_df[data_df[_SPLIT_COL.value] != _SPLIT_VALIDATE_VAL.value]
  train_fname = f'{_LOG_ID.value}.train.${_SPLIT_COL.value}.${_SPLIT_VALIDATE_VAL.value}.csv'
  train_df.to_csv(train_fname, index=False)
  out = subprocess.run(
      [
          'fileutil',
          'cp',
          f'/tmp/{train_fname}',
          # f'gs://{_PROJECT_ID.value}-sight/sight-logs/*{_LOG_ID.value}/{train_fname}'
          f'/cns/oj-d/home/{os.environ["USER"]}/{_LOG_ID.value}/{train_fname}'
      ],
      capture_output=True,
      check=True,
  )

  validate_df = data_df[data_df[_SPLIT_COL.value] != _SPLIT_VALIDATE_VAL.value]
  validate_fname = f'{_LOG_ID.value}.validate.${_SPLIT_COL.value}.${_SPLIT_VALIDATE_VAL.value}.csv'
  validate_df.to_csv(validate_fname, index=False)
  out = subprocess.run(
      [
          'gsutil',
          'cp',
          f'/tmp/{validate_fname}',
          # f'gs://{_PROJECT_ID.value}-sight/sight-logs/*{_LOG_ID.value}/{validate_fname}'
          f'/cns/oj-d/home/{os.environ["USER"]}/{_LOG_ID.value}/{validate_fname}'
      ],
      capture_output=True,
      check=True,
  )

  out = subprocess.run(
      [
          '/google/bin/releases/tunelab/public/ingest_csv',
          '--train_csv_file="${FLAGS_log_prediction_train_path}"',
          '--validation_csv_file="${FLAGS_log_prediction_val_path}"',
          '--col_names="input,target"', '--dataset_name="predictsubsequent"',
          '--output_dir="${FLAGS_basepath}"', 'gsutil', 'cp',
          f'/tmp/{validate_fname}',
          f'gs://{_PROJECT_ID.value}-sight/sight-logs/*{_LOG_ID.value}/{validate_fname}'
      ],
      capture_output=True,
      check=True,
  )


if __name__ == '__main__':
  app.run(main)
