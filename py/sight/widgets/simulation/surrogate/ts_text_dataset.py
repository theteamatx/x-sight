from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import csv
import json
import pandas as pd
from sight.widgets.simulation.surrogate import time_series_to_text

_INFILE_TRAIN = flags.DEFINE_string(
    'infile_train',
    '',
    'The path where the training time series csv file is located.',
)
_INFILE_VALIDATE = flags.DEFINE_string(
    'infile_validate',
    '',
    'The path where the validation time series csv file is located.',
)
_OUTFILE_TRAIN = flags.DEFINE_string(
    'outfile_train',
    '',
    'The path where the output training data for the transformer model file will be written.',
)
_OUTFILE_VALIDATE = flags.DEFINE_string(
    'outfile_validate',
    '',
    'The path where the output validation data for the transformer model file will be written.',
)
_OUTFILE_META = flags.DEFINE_string(
    'outfile_meta',
    '',
    'The path where the output metadata data file will be written.',
)
_HIST_LEN = flags.DEFINE_integer(
    'hist_len',
    5,
    'The path where the output validation data file will be written.',
)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset = time_series_to_text.build_train_val_text_dataset(
    train_ts = pd.read_csv(_INFILE_TRAIN.value),
    validate_ts = pd.read_csv(_INFILE_VALIDATE.value),
    hist_len = _HIST_LEN.value,

  )
  dataset.train.to_csv(_OUTFILE_TRAIN.value, index=False, quoting=csv.QUOTE_ALL)
  dataset.validate.to_csv(_OUTFILE_VALIDATE.value, index=False, quoting=csv.QUOTE_ALL)
  
  with open(_OUTFILE_META.value, 'w') as f:
    json.dump({
      'max_input_len': dataset.max_input_len,
      'max_pred_len': dataset.max_pred_len,
    }, f)


if __name__ == '__main__':
  app.run(main)
