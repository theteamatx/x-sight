"""Loads the simulation records from one or more Sight logs and trains a Gemini transformer on them."""

from absl import app
from absl import flags
from absl import logging
import asyncio
import csv
from dataclasses import dataclass
import json
import os
import pandas as pd
from typing import Optional, Dict, Sequence, Tuple
import uuid

from  sight.widgets.simulation.surrogate import log_to_time_series
from  sight.widgets.simulation.surrogate import time_series_to_text
from  sight.widgets.simulation.surrogate import train_transformer

_LOG_ID = flags.DEFINE_list(
    'log_id',
    [],
    'Unique IDs of the simulation run logs on which to train the Gemini model.',
)
_WORKDIR = flags.DEFINE_string(
    'workdir',
    '/tmp',
    'Path of the directory where intermediate files should be stored.',
)
_PROJECT_ID = flags.DEFINE_string(
    'project_id', os.environ['PROJECT_ID'], 'ID of the current GCP project.'
)
_SPLIT = flags.DEFINE_enum('split', 
                              'random', 
                              ['random', 'lat', 'lng'], 
                              'The algorithm for splitting time series into training and validation.')
_TRAIN_FRAC = flags.DEFINE_float(
    'train_frac',
    .8,
    'The path where the output validation data file will be written.',
)
_TRAIN_MIN_VAL = flags.DEFINE_float(
    'train_min_val',
    0,
    'The0 minimum value (inclusive) of the training split selection column to include in the training set.',
)
_TRAIN_MAX_VAL = flags.DEFINE_float(
    'train_max_val',
    100,
    'The maximum value (inclusive) of the training split selection column to include in the training set.',
)
_ENCODING = flags.DEFINE_enum('encoding', 
                              'init_recent_hist', 
                              ['init_recent_hist', 'init_bound_to_auto'], 
                              'The algorithm for encoding time series into transformer text.')
_HIST_LEN = flags.DEFINE_integer(
    'hist_len',
    5,
    'The path where the output validation data file will be written.',
)
_DATA_CELL = flags.DEFINE_string(
    'data_cell', 'oj', 'Borg cell where training data will be stored.'
)


_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '',
    'If we need to finetune an existing model, this is he path where the model\'s checkpoint file is stored.',
)
_FAMILY = flags.DEFINE_enum(
    'family',
    'gemini',
    ['gemini', 'ulm'],
    'The major model family.',
)
_VARIANT = flags.DEFINE_string(
    'variant',
    'GEMINI-XXS',
    'The variant of the model family.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    128,
    'The batch size of the training process.',
)
_NUM_TRAIN_STEPS = flags.DEFINE_integer(
    'num_train_steps',
    100000,
    'The number of training steps.',
)
_NUM_EVAL_STEPS = flags.DEFINE_integer(
    'num_eval_steps',
    2000,
    'The number of evaluation steps.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    1e-7,
    'The learning rate of the training process.',
)
_SAVE_INTERVAL_STEPS = flags.DEFINE_integer(
    'save_interval_steps',
    2000,
    'The number of checkpoints to save.',
)
_TRAIN_CELL = flags.DEFINE_string(
    'train_cell',
    'gl',
    'The cell where the training job will execute.',
)
_PLATFORM_NAME = flags.DEFINE_string(
    'platform_name',
    'pf',
    'The name configuration of the hardware compute platform that will perform the training.',
)
_PLATFORM_MESH = flags.DEFINE_string(
    'platform_mesh',
    '2x2x4',
    'The mesh of the configuration of the hardware compute platform that will perform the training.',
)
_PRIORITY = flags.DEFINE_integer(
    'priority',
    200,
    'The execution priority of the training job.',
)
_MESH = flags.DEFINE_string(
    'mesh',
    '(1,16,1)',
    'The configuration of the communication/parallelization mesh.',
)

@dataclass
class LoadedDataset:
  train: pd.DataFrame
  validate: pd.DataFrame
  max_input_len: int
  max_pred_len: int


async def load_dataset(log_id: str, split_label: str, encoding_label: str, run_id: str) -> LoadedDataset:
  """Loads a time series dataset from a given Sight log.
  
  Arguments:
    log_id: UID of the Sight log from which the simulations are being loaded.
    split_label: Human-readable label that captures the details of the algorithm for splitting the time series
      dataset into training and validation subsets.
    encoding_label: Human-readable label that captures the details of the text encoding of the time series.
    run_ud: UID of this data loading run, used to make consistent unique temporary files.
  
  Returns:
    Record of the loaded dataset.
  """
  print(f'Loading TS Dataset from {log_id}')

  # Load the time series DataFrame
  ts_file_path = os.path.join(_WORKDIR.value, f'ts.{log_id}.csv')
  if os.path.exists(ts_file_path):
    ts = pd.read_csv(ts_file_path)
  else:
    ts = log_to_time_series.load_ts(
      log_id,
      _PROJECT_ID.value,
    )
    ts.reset_index().to_csv(
        os.path.join(_WORKDIR.value, f'ts.{log_id}.csv'), index=False)
  print(ts)

  print(f'Splitting TS Dataset from {log_id} into Train and Validate')

  if _SPLIT.value == 'random':
    ts_dataset = log_to_time_series.split_time_series_randomly(ts, _TRAIN_FRAC.value)
  elif _SPLIT.value == 'lat':
    ts_dataset = log_to_time_series.split_time_series_by_condition(ts, 'initial:lat', lambda x: x>=_TRAIN_MIN_VAL.value and x<=_TRAIN_MAX_VAL.value)
  elif _SPLIT.value == 'lng':
    ts_dataset = log_to_time_series.split_time_series_by_condition(ts, 'initial:lng', lambda x: x>=_TRAIN_MIN_VAL.value and x<=_TRAIN_MAX_VAL.value)

  ts_dataset.train.reset_index().to_csv(
    os.path.join(_WORKDIR.value, f'ts.{log_id}.train.{split_label}.{run_id}.csv'), index=False)
  ts_dataset.validate.reset_index().to_csv(
    os.path.join(_WORKDIR.value, f'ts.{log_id}.validate.{split_label}.{run_id}.csv'), index=False)
  
  print(f'Building Transformer Text Dataset from TS, for log {log_id}')
  if _ENCODING.value == 'init_recent_hist':
    transformer_dataset = time_series_to_text.build_recent_hist_text_dataset(
      train_ts = ts_dataset.train.reset_index(),
      validate_ts = ts_dataset.validate.reset_index(),
      hist_len = _HIST_LEN.value,
    )
  elif _ENCODING.value == 'init_bound_to_auto':
    transformer_dataset = time_series_to_text.build_train_val_text_dataset_init_bound_to_auto(
      train_ts = ts_dataset.train.reset_index(),
      validate_ts = ts_dataset.validate.reset_index(),
    )
  
  transformer_dataset.train.to_csv(
    os.path.join(_WORKDIR.value, f'transformer.{log_id}.train.{split_label}.{encoding_label}.{run_id}.csv'), 
    index=False, quoting=csv.QUOTE_ALL)
  transformer_dataset.validate.to_csv(
    os.path.join(_WORKDIR.value, f'transformer.{log_id}.validate.{split_label}.{encoding_label}.{run_id}.csv'), 
    index=False, quoting=csv.QUOTE_ALL)

  # with open(os.path.join(_WORKDIR.value, f'transformer.{log_id}.meta.{split_label}.{encoding_label}.{run_id}.csv'), 'w') as f:
  #   json.dump({
  #     'max_input_len': transformer_dataset.max_input_len,
  #     'max_pred_len': transformer_dataset.max_pred_len,
  #   }, f)
  
  return LoadedDataset(
    transformer_dataset.train,
    transformer_dataset.validate,
    transformer_dataset.max_input_len,
    transformer_dataset.max_pred_len,
  )

  

async def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _SPLIT.value == 'random':
    split_label = f'{_SPLIT.value}_tf_{_TRAIN_FRAC.value}'
  elif _SPLIT.value == 'lat' or _SPLIT.value == 'lng':
    split_label = f'{_SPLIT.value}_min_{_TRAIN_MIN_VAL.value}_max_{_TRAIN_MAX_VAL.value}'.replace('-', '_')

  if _ENCODING.value == 'init_recent_hist':
    encoding_label = f'{_ENCODING.value}_hist_{_HIST_LEN.value}'
  elif _ENCODING.value == 'init_bound_to_auto':
    encoding_label = _ENCODING.value

  run_id = str(uuid.uuid4()).replace('-', '_')

  # Load all the log files
  load_tasks = []
  for log_id in _LOG_ID.value:
    load_tasks.append(asyncio.create_task(load_dataset(log_id, run_id, split_label, encoding_label)))

  all_train = []
  all_validate = []
  max_input_len = 0
  max_pred_len = 0
  for dataset in await asyncio.gather(*load_tasks):
    all_train.append(dataset.train)
    all_validate.append(dataset.validate)
    max_input_len = max(max_input_len, dataset.max_input_len)
    max_pred_len = max(max_pred_len, dataset.max_pred_len)
  
  train = pd.concat(all_train, axis=0)
  validate = pd.concat(all_train, axis=0)


  all_log_ids_label = ','.join([str(log_id) for log_id in _LOG_ID.value]) + f'_{split_label}_{encoding_label}_{run_id}'
  print(f'Ingesting full Transformer text dataset.')
  ingested_dataset = train_transformer.ingest_transformer_dataset(
    train,
    validate,
    _WORKDIR.value,
    all_log_ids_label,
    _DATA_CELL.value
  )

  logging.info ('max_input_len=%s, max_pred_len=%s', max_input_len, max_pred_len)

  print(f'Launching Gemini to train on Transformer text dataset.')
  train_transformer.finetune_gemini(
    dataset = ingested_dataset,
    max_input_len = max_input_len,
    max_pred_len = max_pred_len,
    work_dir = _WORKDIR.value,
    label = all_log_ids_label,
    checkpoint_path = _CHECKPOINT_PATH.value,
    family = _FAMILY.value,
    variant = _VARIANT.value,
    batch_size = _BATCH_SIZE.value,
    num_train_steps = _NUM_TRAIN_STEPS.value,
    num_eval_steps = _NUM_EVAL_STEPS.value,
    learning_rate = _LEARNING_RATE.value,
    save_interval_steps = _SAVE_INTERVAL_STEPS.value,
    cell = _TRAIN_CELL.value,
    platform_name = _PLATFORM_NAME.value,
    platform_mesh = _PLATFORM_MESH.value,
    priority = _PRIORITY.value,
    mesh = _MESH.value
  )
  
def main_wrapper(argv: Sequence[str]):
  asyncio.run(main(argv))


if __name__ == '__main__':
  app.run(main_wrapper)
