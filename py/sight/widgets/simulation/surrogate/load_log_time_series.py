"""Queries the Simulation entries in a Sight log and saves them as a time series."""


from absl import app
from absl import flags
import os
from typing import Optional, Dict, Sequence, Tuple

from  sight.widgets.simulation.surrogate import log_to_time_series

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
_OUTFILE_TRAIN = flags.DEFINE_string(
    'outfile_train',
    '',
    'Path of the file where the training subset of the run simulation time series will be stored.',
)
_OUTFILE_VALIDATE = flags.DEFINE_string(
    'outfile_validate',
    '',
    'The path where the output validation data for the transformer model file will be written.',
)
_PROJECT_ID = flags.DEFINE_string(
    'project_id', os.environ['PROJECT_ID'], 'ID of the current GCP project.'
)

_TRAIN_FRAC = flags.DEFINE_float(
    'train_frac',
    .8,
    'The path where the output validation data file will be written.',
)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _OUTFILE.value and os.path.exists(_OUTFILE.value):
    ts = pd.read_csv(_OUTFILE.value) 
  else:
    ts = log_to_time_series.load_ts(
      _LOG_ID.value,
      _PROJECT_ID.value,
    )
  
  print(ts)

  if _OUTFILE.value:
    ts.reset_index().to_csv(_OUTFILE.value, index=False)
    
    if _OUTFILE_TRAIN.value or _OUTFILE_VALIDATE.value:
      dataset = log_to_time_series.split_time_series(ts, _TRAIN_FRAC.value)
      if _OUTFILE_TRAIN.value:
        dataset.train.reset_index().to_csv(_OUTFILE_TRAIN.value, index=False)
      if _OUTFILE_VALIDATE.value:
        dataset.validate.reset_index().to_csv(_OUTFILE_VALIDATE.value, index=False)

if __name__ == '__main__':
  app.run(main)
