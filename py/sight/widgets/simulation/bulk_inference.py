from datetime import datetime
import json
import os
import subprocess
from typing import Sequence, Tuple

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging

_LOG_ID = flags.DEFINE_string(
    'log_id',
    '',
    'Unique ID of the simulation run log.',
)
_INPUT_META_PATH = flags.DEFINE_string(
    'input_meta_path',
    '',
    'The path where the output metadata data file is stored.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    128,
    'The batch size of the training process.',
)
_CELL = flags.DEFINE_string(
    'cell',
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
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '',
    'If we need to finetune an existing model, this is he path where the model\'s checkpoint file is stored.',
)
_TRAINER_XID = flags.DEFINE_string(
    'trainer_xid',
    '',
    'The ID of the XManager job that trained the model being used for inference.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(_INPUT_META_PATH.value) as f:
    meta = json.load(f)
    max_input_len = meta['max_input_len']
    max_pred_len = meta['max_pred_len']

  cmd = [
      '/google/bin/releases/tunelab/public/bulk_inference_jax_on_beam',
      f'--input_spec=arrayrecord:/cns/oj-d/home/bronevet/kokua/experiments/bronevet/dataset/simulation_transformer_{_LOG_ID.value}/simulation_transformer_{_LOG_ID.value}/validation/simulation_transformer_{_LOG_ID.value}.array_record-00000-of-00001',
      f'--output_spec=arrayrecord:/cns/oj-d/home/bronevet/kokua/experiments/bronevet/dataset/simulation_transformer_{_LOG_ID.value}/simulation_transformer_{_LOG_ID.value}/validation/predictions/model_output.recordio@*',
      f'--batch_size={_BATCH_SIZE.value}',
      '--extra_inputs=EXTRA_INPUTS:{\'temperature\': 0.0}',
      f'--extra_inputs=INPUT_SEQ_LEN:{max_input_len}',
      f'--extra_inputs=MAX_DECODE_STEPS:{max_pred_len}',
      '--prompt_feature_name=input',
      f'--accelerator_priority_range={_PRIORITY.value},{_PRIORITY.value}',
      f'--flume_priority={_PRIORITY.value}',
      # '--xm_resource_alloc=x/early-pipeline-alloc',
      # '--xm_resource_pool=x',
      '--run_on_xm',
      f'--flume_borg_cells={_CELL.value}',
      f'--tpu_borg_cell={_CELL.value}',
      # '--charged_alloc=group:x/early-pipeline-alloc',
      f'--platform={_PLATFORM_NAME.value}',
      f'--topology={_PLATFORM_MESH.value}',
      f'--ici_mesh_shape="{_MESH.value}"',
  ]
  if _CHECKPOINT_PATH.value:
    cmd.append(f'--model_checkpoint={_CHECKPOINT_PATH.value}')
  elif _TRAINER_XID.value:
    cmd.append(f'--trainer_xid={_TRAINER_XID.value}')
  print(' '.join(cmd))
  out = subprocess.run(
      cmd,
      capture_output=True,
      check=True,
  )
  print(out)


if __name__ == '__main__':
  app.run(main)
