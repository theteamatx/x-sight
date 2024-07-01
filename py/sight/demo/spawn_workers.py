# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary to spawn multiple workers with given file."""

import os
import math
import subprocess
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple

from absl import app
from absl import flags
from absl import logging
from sight.proto import sight_pb2
from sight.sight import Sight

FLAGS = flags.FLAGS
_file_name = 'spawn_worker.py'
_SERVICE_ID = 'default'

def get_sight_instance():
  print('creating sight object')
  params = sight_pb2.Params(
      label='original_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj

def _get_experiment_name(sight: Any) -> str:
  return (
        'Sight_Kokua_Study_'
        + sight.params.label.replace(' ', '_')
        + '_'
        + str(sight.id)
        + '_'
        + datetime.now().strftime('%Y%m%d_%H%M%S')
    )

def start_jobs_in_dsub(
    num_train_workers: int,
    num_trials: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image,
    deployment_mode: str,
    worker_mode: str,
    sight: Any,
):
  method_name = 'start_jobs'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  # print('num_train_workers : ', num_train_workers)
  # print('num_trials : ', num_trials)
  # print('binary_path : ', binary_path)
  # print('optimizer_type : ', optimizer_type)
  # print('docker_image : ', docker_image)
  # print('deployment_mode : ', deployment_mode)
  # print('worker_mode : ', worker_mode)

  sight.enter_block('Worker Spawning', sight_pb2.Object())
  with open('/tmp/optimization_tasks.tsv', 'w') as outf:
    outf.write('--env worker_id\t--env num_samples\t--env worker_location\n')
    num_tasks_per_worker = math.floor(num_trials / num_train_workers)
    for worker_id in range(num_train_workers):
      tasks_for_cur_worker = num_tasks_per_worker
      # If _NUM_TRIALS is not evenly divisible by num_train_workers, add
      # the extra extra tasks to the first few workers.
      if worker_id < num_trials % num_train_workers:
        tasks_for_cur_worker += 1
      outf.write(f'{worker_id}\t{tasks_for_cur_worker}\t{sight.location}\n')
      sight.location.next()


  remote_script = (
      # 'gs://dsub_cameltrain/cameltrain/' + binary_path.split('/')[-1]
      f'gs://{os.environ["PROJECT_ID"]}-sight/d-sub/binary/' + binary_path.split('/')[-1]
  )
  print(f'Uploading {binary_path}...')
  subprocess.run(['gsutil', 'cp', '-c', binary_path, remote_script], check=True)

  if not FLAGS.service_account:
    raise ValueError(
        'flag --service_account required for worker_mode as dsub_cloud_worker.'
    )

  command = (
      'ls -l && echo "${SCRIPT}" && echo "${PYTHONPATH}" && python3 "${SCRIPT}"'
      # 'ls -l && echo "${SCRIPT}" && echo "${PYTHONPATH}" && python3 "${SCRIPT}"'
      # + f' --decision_mode={decision_mode}'
      + f' --deployment_mode={deployment_mode}'
      + f' --worker_mode={worker_mode}'
      + f' --optimizer_type={optimizer_type}'
      + f' --project_id={os.environ["PROJECT_ID"]}'
  )

  logging_path = f'gs://{os.environ["PROJECT_ID"]}-sight/d-sub/portfolio/logs/'
  logging_path += str(sight.id)


  print('sight.id=%s' % sight.id)
  args = [
      'dsub',
      '--provider=google-cls-v2',
      f'--regions={FLAGS.project_region}',
      # f'--location={_PROJECT_REGION.value}',
      f'--image={docker_image}',
      f'--machine-type={FLAGS.dsub_machine_type}',
      f'--project={FLAGS.project_id}',
      # f'--logging=gs://{os.environ["PROJECT_ID"]}-sight/d-sub/logs/{service._SERVICE_ID}/{sight.id}',
      f'--logging={logging_path}',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      # '--env',
      # 'PYTHONPATH=/project',
      '--env',
      f'SIGHT_SERVICE_ID={_SERVICE_ID}',
      '--input',
      f'SCRIPT={remote_script}',
      f'--command={command}',
      f'--service-account={FLAGS.service_account}@{os.environ["PROJECT_ID"]}.iam.gserviceaccount.com',
      f'--boot-disk-size={FLAGS.dsub_boot_disk_size}',
      '--tasks',
      '/tmp/optimization_tasks.tsv',
      '--name',
      _get_experiment_name(sight)[:63],
  ]

  logging.info('CLI=%s', ' '.join(args))
  subprocess.run(args, check=True)

  sight.exit_block('Worker Spawning', sight_pb2.Object())
  logging.info('worker logs available at : %s', f'gs://{os.environ["PROJECT_ID"]}/d-sub/logs/default')
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    start_jobs_in_dsub(
              num_train_workers=1,
              num_trials=1,
              binary_path='kokua-worker.py',
              optimizer_type='dummy',
              docker_image='gcr.io/cameltrain/sight-portfolio-worker',
              deployment_mode='worker_mode',
              worker_mode='dsub_cloud_worker',
              sight=sight,
          )

if __name__ == "__main__":
  app.run(main)
