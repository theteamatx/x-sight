"""Functionality for training transformer models."""

from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
import pandas as pd
import subprocess


@dataclass
class IngestedDataset:
  dataset_name: str
  base_dir_path: str 
  train_path: str
  validate_path: str

def ingest_transformer_dataset(train: pd.DataFrame,
                               validate: pd.DataFrame,
                               work_dir: str,
                               label: str,
                               data_cell: str,
                               ):
  local_train_path = os.path.join(work_dir, f'transformer.{label}.train.csv')
  train.to_csv(
    local_train_path, 
    index=False, quoting=csv.QUOTE_ALL)
  
  local_validate_path = os.path.join(work_dir, f'transformer.{label}.validate.csv')
  validate.to_csv(
    local_validate_path, 
    index=False, quoting=csv.QUOTE_ALL)
  
  base_dir_path = os.path.join('/cns', f'{data_cell}-d', 'home', os.environ["USER"], 'dataset')
  print('base_dir_path=', base_dir_path)
  dataset_name = f'simulation_transformer_{label}'
  ingest_cmd = [
    '/google/bin/releases/tunelab/public/ingest_csv',
    f'--train_csv_file={local_train_path}',
    f'--validation_csv_file={local_validate_path}',
    f'--col_names=input,pred',
    f'--dataset_name={dataset_name}', 
    f'--output_dir={base_dir_path}',
    '--overwrite',
  ]
  out = subprocess.run(
    ingest_cmd,
    stdout = subprocess.PIPE,
    stderr = subprocess.STDOUT,
    text=True,
    # check=True,
  )
  print(out.stdout)

  return IngestedDataset(
    dataset_name,
    base_dir_path,
    os.path.join(base_dir_path, f'simulation_transformer_{label}', 'train', f'{dataset_name}.array_record-00000-of-00001'),
    os.path.join(base_dir_path, f'simulation_transformer_{label}', 'validation', f'{dataset_name}.array_record-00000-of-00001'),
  )

def finetune_gemini(
    dataset: IngestedDataset,
    max_input_len: int,
    max_pred_len: int,
    work_dir: str,
    label: str,
    checkpoint_path: str = '',
    family: str = 'gemini',
    variant: str = 'GEMINI-XXS',
    batch_size: int = 128,
    num_train_steps: int = 100000,
    num_eval_steps: int = 2000,
    learning_rate: float = 1e-7,
    save_interval_steps: int = 2000,
    cell: str = 'gl',
    platform_name: str = 'pf',
    platform_mesh: str = '2x2x4',
    priority: int = 200,
    mesh: str = '(1,16,1)'
  ):
  mixtures_path = os.path.join(work_dir, f'mixtures.{label}.textproto')
  with open(mixtures_path, 'w') as f:
    f.write(f"""
# proto-file: google3/learning/language/tunelab/tunekit/api/common/proto/task.proto
# proto-message: Task

# proto that defines training tasks
# fed from ingestion.sh

train_datasets {{
  name: "simulation_transformer_{label}"
  data_path: "arrayrecord:{dataset.train_path}"
  text_feature_keys: "input"
  label_key: "pred"
}}
eval_datasets {{
  name: "simulation_transformer_{label}"
  data_path: "arrayrecord:{dataset.validate_path}"
  text_feature_keys: "input"
  label_key: "pred"
}}
            """)
    
  date = datetime.today().strftime('%Y-%m-%d.%H:%M:%S')

  cmd = [
        '/google/bin/releases/tunelab/public/finetune', 
        f'--family={family}',
        f'--variant={variant}',
        f'--task_proto_data_path={mixtures_path}',
        f'--train_dataset_name={dataset.dataset_name}',
        f'--eval_dataset_name={dataset.dataset_name}',
        f'--output_dir={dataset.base_dir_path}/{variant}/{date}',
        f'--batch_size={batch_size}',
        f'--inputs_length={max_input_len}',
        f'--targets_length={max_pred_len}',
        f'--num_train_steps={num_train_steps}',
        f'--eval_interval_steps={num_eval_steps}',
        f'--save_interval_steps={save_interval_steps}',
        f'--learning_rate={learning_rate}',
        f'--xm_resource_alloc=x/early-pipeline-alloc',
        f'--xm_resource_pool=x',
        f'--priority={priority}',
        f'--cell={cell}',
        f'--platform={platform_name}_{platform_mesh}',
        f'--mesh={mesh}',
      ]
  if checkpoint_path:
    cmd.append(f'--checkpoint_path={checkpoint_path}')
  print(' '.join(cmd))
  out = subprocess.run(cmd,
      capture_output=True,
      # check=True,
  )
  print(out)

