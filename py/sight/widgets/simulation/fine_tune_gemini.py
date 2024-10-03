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
_BASE_OUTPUT_PATH = flags.DEFINE_string(
    'base_output_path',
    f'/cns/oj-d/home/${os.environ["USER"]}/kokua/',
    'The base path of the output directory structure.',
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


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  date = datetime.today().strftime('%Y-%m-%d.%H:%M:%S')
  dataset_id = f'simulation_transformer_{_LOG_ID.value}'
  output_path = f'{_BASE_OUTPUT_PATH.value}/experiments/{os.environ["USER"]}/{dataset_id}/{_VARIANT.value}/{date}'

  with open(_INPUT_META_PATH.value) as f:
    meta = json.load(f)
    max_input_len = meta['max_input_len']
    max_pred_len = meta['max_pred_len']

  with open('/tmp/mixtures.textproto', 'w') as f:
    f.write(f"""
# proto-file: google3/learning/language/tunelab/tunekit/api/common/proto/task.proto
# proto-message: Task

# proto that defines training tasks
# fed from ingestion.sh

train_datasets {{
  name: "simulation_transformer_{_LOG_ID.value}"
  data_path: "arrayrecord:/cns/oj-d/home/bronevet/kokua/experiments/bronevet/dataset/simulation_transformer_{_LOG_ID.value}/simulation_transformer_{_LOG_ID.value}/train/simulation_transformer_{_LOG_ID.value}.array_record-00000-of-00001"
  text_feature_keys: "input"
  label_key: "pred"
}}
eval_datasets {{
  name: "simulation_transformer_{_LOG_ID.value}"
  data_path: "arrayrecord:/cns/oj-d/home/bronevet/kokua/experiments/bronevet/dataset/simulation_transformer_{_LOG_ID.value}/simulation_transformer_{_LOG_ID.value}/validation/simulation_transformer_{_LOG_ID.value}.array_record-00000-of-00001"
  text_feature_keys: "input"
  label_key: "pred"
}}
            """)
  cmd = [
      '/google/bin/releases/tunelab/public/finetune',
      f'--family={_FAMILY.value}',
      f'--variant={_VARIANT.value}',
      f'--task_proto_data_path=/tmp/mixtures.textproto',
      f'--train_dataset_name={dataset_id}',
      f'--eval_dataset_name={dataset_id}',
      f'--output_dir={output_path}',
      f'--batch_size={_BATCH_SIZE.value}',
      f'--inputs_length={max_input_len}',
      f'--targets_length={max_pred_len}',
      f'--num_train_steps={_NUM_TRAIN_STEPS.value}',
      f'--eval_interval_steps={_NUM_EVAL_STEPS.value}',
      f'--save_interval_steps={_SAVE_INTERVAL_STEPS.value}',
      f'--learning_rate={_LEARNING_RATE.value}',
      f'--xm_resource_alloc=x/early-pipeline-alloc',
      f'--xm_resource_pool=x',
      f'--priority={_PRIORITY.value}',
      f'--learning_rate=0.0005',
      f'--cell={_CELL.value}',
      f'--platform={_PLATFORM_NAME.value}_{_PLATFORM_MESH.value}',
      f'--mesh={_MESH.value}',
  ]
  if _CHECKPOINT_PATH.value:
    cmd.append(f'--checkpoint_path={_CHECKPOINT_PATH.value}')
  print(' '.join(cmd))
  out = subprocess.run(
      cmd,
      capture_output=True,
      #   check=True,
  )
  print(out)


if __name__ == '__main__':
  app.run(main)
