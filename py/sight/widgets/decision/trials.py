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
"""Functionality for using Vizier to drive decisions."""

from datetime import datetime
import math
import os
import random
import subprocess
import time
from typing import Any, Dict, Optional

from absl import flags
from dotenv import load_dotenv
import grpc
from helpers.logs.logs_handler import logger as logging
import pytz
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.widgets.decision import decision
from sight.widgets.decision import utils
# from sight.widgets.decision.acme import acme_optimizer_client
from sight.widgets.decision.optimizer_client import OptimizerClient
from sight_service.proto import service_pb2

load_dotenv()

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'The name of the experiment this worker will participate in.',
)
_PROJECT_ID = flags.DEFINE_string(
    'project_id',
    os.environ.get('PROJECT_ID', os.environ.get('GOOGLE_CLOUD_PROJECT', '')),
    'Id of cloud project')
_PROJECT_REGION = flags.DEFINE_string('project_region', 'us-central1',
                                      'location to store project-data')
_DSUB_MACHINE_TYPE = flags.DEFINE_string('dsub_machine_type', 'e2-standard-2',
                                         '')

_DSUB_BOOT_DISK_SIZE = flags.DEFINE_integer('dsub_boot_disk_size', 30, '')

_file_name = 'trials.py'
FLAGS = flags.FLAGS


def _get_experiment_name(sight: Any) -> str:
  if _EXPERIMENT_NAME.value:
    return _EXPERIMENT_NAME.value
  else:
    return ('Sight_Decision_Study_' + sight.params.label.replace(' ', '_') +
            '_' + str(sight.id) + '_' +
            datetime.now().strftime('%Y%m%d_%H%M%S'))


def launch(
    decision_configuration: sight_pb2.DecisionConfigurationStart,
    sight: Any,
):
  """Launches the experiment with the service.

  Args:
    decision_configuration: Object containing attributes for the question to be launched
    sight: The Sight object to be used for logging.
  """
  method_name = 'launch'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  req = service_pb2.LaunchRequest()
  req.decision_config_params.CopyFrom(decision_configuration)
  req.label = sight.params.label
  req.client_id = str(sight.id)
  req.question_label = decision_configuration.question_label

  response = service.call(lambda s, meta: s.Launch(req, 300, metadata=meta))
  logging.info('##### Launch response=%s #####', response)

  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def start_worker_jobs(sight, 
                      optimizer_config: dict, 
                      worker_configs:dict, 
                      optimizer_type: str):
  # for worker_name in optimizer_config['worker_names']:
  #   worker_details = worker_configs[worker_name]

  num_questions = optimizer_config['num_questions']
  for worker, worker_count in optimizer_config['workers'].items():
    # print('worker_count : ', worker_count)
    worker_details = worker_configs[worker]
    start_jobs(worker_count, 
               worker_details['binary'], 
               optimizer_type,
               worker_details['docker'], 
               'train', 
               'worker_mode',
               optimizer_config['mode'], 
               FLAGS.cache_mode,
               sight)


def append_ist_time_to_logging_path_12hr():
  # Define IST timezone
  ist = pytz.timezone('Asia/Kolkata')
  # Get the current date and time in IST
  current_time = datetime.now(ist)
  formatted_time = current_time.strftime('%Y-%m-%d-%I-%M-%S')
  return formatted_time


def start_job_in_docker(
    num_trials: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image: str,
    decision_mode: str,
    server_mode: str,
    worker_mode: str,
    decision_params: str,
    sight: Any,
):
  """Starts a single worker in a docker container.

  Args:
    num_trials: The number of times the experiment will be run during training.
    binary_path: Path of the Blaze target of this binary.
    optimizer_type: Type of optimizer we are using.
    docker_image: Path of the docker image within which the binary is to be run.
    decision_mode: add
    server_mode: add
    worker_mode: add
    decision_params: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_job_in_docker'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  sight.enter_block('Worker Spawning', sight_pb2.Object())
  # Write the script that will execute the binary within the docker container.
  decision_params_arg = (f' --decision_params={decision_params}'
                         if decision_params else '')
  os.makedirs('/tmp/sight_script', exist_ok=True)
  with open('/tmp/sight_script/sight_decision_command.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('echo "$PYTHONPATH"')
    f.write(
        '/usr/bin/python3'
        f' /project/{binary_path.split("/")[-1]} --decision_mode={decision_mode} --server_mode={server_mode}'
        f' --worker_mode={worker_mode} --optimizer_type={optimizer_type} --num_trials={num_trials} '
    )
    if FLAGS.service_account:
      f.write(f' --service_account={FLAGS.service_account}')
    f.write(f' {decision_params_arg}\n ')
  os.chmod('/tmp/sight_script/sight_decision_command.sh', 0o755)
  subprocess.run(['cp', binary_path, '/tmp'], check=True)

  args = [
      'docker',
      'run',
      '-v',
      f'/tmp/{binary_path.split("/")[-1]}:/project/{binary_path.split("/")[-1]}:ro',
      '-v',
      '/tmp/sight_script:/project/sight_script:ro',
      '-v',
      f'{FLAGS.gcloud_dir_path}:/project/.config/gcloud:ro',
      '--env',
      'GOOGLE_APPLICATION_CREDENTIALS=/project/.config/gcloud/application_default_credentials.json',
      '--env',
      f'GOOGLE_CLOUD_PROJECT={_PROJECT_ID.value}',
      '--env',
      f'PROJECT_ID={os.environ["PROJECT_ID"]}',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
      '--env',
      f'worker_location={sight.location.get()}',
      '--env',
      f'num_samples={num_trials}',
      '--net=host',
      '-t',
      '-i',
      '--rm',
      docker_image,
      '/project/sight_script/sight_decision_command.sh',
      # 'bash',
  ]
  logging.info('DOCKER CONTAINER SPAWNING =%s', ' '.join(args))
  subprocess.run(args, check=True)

  sight.exit_block('Worker Spawning', sight_pb2.Object())
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def start_jobs(num_train_workers: int, binary_path: Optional[str],
               optimizer_type: str, docker_image, decision_mode: str,
               server_mode: str, worker_mode: str, cache_mode: str, sight: Any):
  """Starts the dsub workers that will run the optimization.

  Args:
    num_train_workers: Number of workers to use in a training run.
    binary_path: Path of the Blaze target of this binary.
    optimizer_type: Type of optimizer we are using.
    docker_image: Path of the docker image within which the binary is to be run.
    decision_mode: add
    server_mode: add
    worker_mode: add
    cache_mode: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_jobs'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  sight.enter_block('Worker Spawning', sight_pb2.Object())
  with open('/tmp/optimization_tasks.tsv', 'w') as outf:
    outf.write('--env worker_id\t--env worker_location\n')
    # num_tasks_per_worker = math.floor(num_trials / num_train_workers)
    for worker_id in range(num_train_workers):
      # tasks_for_cur_worker = num_tasks_per_worker
      # # If _NUM_TRIALS is not evenly divisible by num_train_workers, add
      # # the extra extra tasks to the first few workers.
      # if worker_id < num_trials % num_train_workers:
      #   tasks_for_cur_worker += 1
      outf.write(f'{worker_id}\t{sight.location.get()}\n')
      sight.location.get().next()

  remote_script = (
      # 'gs://dsub_cameltrain/cameltrain/' + binary_path.split('/')[-1]
      f'gs://{os.environ["PROJECT_ID"]}-sight/d-sub/binary/{str(sight.id)}/' +
      binary_path.split('/')[-1])
  print(f'Uploading {binary_path}...')
  subprocess.run(['gsutil', 'cp', '-c', binary_path, remote_script], check=True)

  if not FLAGS.service_account:
    raise ValueError(
        'flag --service_account required for worker_mode as dsub_cloud_worker.')

  command = (
      'ls -l && echo "${SCRIPT}" && echo "${PYTHONPATH}" && python3 "${SCRIPT}"'
      + f' --decision_mode={decision_mode}' + f' --server_mode={server_mode}' +
      f' --worker_mode={worker_mode}' + f' --optimizer_type={optimizer_type}' +
      f' --cache_mode={cache_mode}')
  if FLAGS.env_name:
    command += f' --env_name={FLAGS.env_name}'

  logging_path = f'gs://{os.environ["PROJECT_ID"]}-sight/d-sub/logs/{sight.params.label}/{append_ist_time_to_logging_path_12hr()}/'
  if (FLAGS.parent_id):
    logging_path += f'{FLAGS.parent_id}/'
  logging_path += str(sight.id)

  env_vars = [
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      f'PORT={service.get_port_number()}',
      f'PROJECT_ID={os.environ["PROJECT_ID"]}',
  ]

  print("FLAGS.server_mode : ", FLAGS.server_mode)
  if FLAGS.server_mode == 'vm':
    if FLAGS.ip_addr == 'localhost':
      raise ValueError("ip_address must be provided for workers")
    env_vars += ['--env', f'IP_ADDR={FLAGS.ip_addr}']
  elif FLAGS.server_mode == 'cloud_run':
    env_vars += ['--env', f'SIGHT_SERVICE_ID={service._SERVICE_ID}']

  print('sight.id=%s' % sight.id)
  args = [
      'dsub',
      '--provider=google-cls-v2',
      f'--regions={_PROJECT_REGION.value}',
      '--use-private-address',
      f'--image={docker_image}',
      f'--machine-type={_DSUB_MACHINE_TYPE.value}',
      f'--project={_PROJECT_ID.value}',
      f'--logging={logging_path}',
      *env_vars,
      '--input',
      f'SCRIPT={remote_script}',
      f'--command={command}',
      f'--service-account={FLAGS.service_account}@{os.environ["PROJECT_ID"]}.iam.gserviceaccount.com',
      f'--boot-disk-size={_DSUB_BOOT_DISK_SIZE.value}',
      '--tasks',
      '/tmp/optimization_tasks.tsv',
      '--name',
      _get_experiment_name(sight)[:63],
  ]

  logging.info('CLI=%s', ' '.join(args))
  subprocess.run(args, check=True)

  sight.exit_block('Worker Spawning', sight_pb2.Object())
  logging.info('worker logs available at : %s',
               f'gs://{os.environ["PROJECT_ID"]}/d-sub/logs/default')
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def start_job_in_dsub_local(
    num_train_workers: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image,
    decision_mode: str,
    server_mode: str,
    worker_mode: str,
    cache_mode: str,
    sight: Any,
):
  """Starts the dsub workers that will run the optimization.

  Args:
    num_train_workers: Number of workers to use in a training run.
    num_trials: The number of times the experiment will be run during training.
    binary_path: Path of the Blaze target of this binary.
    optimizer_type: Type of optimizer we are using.
    docker_image: Path of the docker image within which the binary is to be run.
    decision_mode: add
    server_mode: add
    worker_mode: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_job_in_dsub_local'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  sight.enter_block('Worker Spawning locally', sight_pb2.Object())
  with open('/tmp/optimization_tasks.tsv', 'w') as outf:
    # outf.write('--env worker_id\t--env num_samples\t--env worker_location\n')
    outf.write('--env worker_id\t--env worker_location\n')
    # num_tasks_per_worker = math.floor(num_trials / num_train_workers)
    for worker_id in range(num_train_workers):
      # tasks_for_cur_worker = num_tasks_per_worker
      # # If _NUM_TRIALS is not evenly divisible by num_train_workers, add
      # # the extra extra tasks to the first few workers.
      # if worker_id < num_trials % num_train_workers:
      #   tasks_for_cur_worker += 1
      # outf.write(
      #     f'{worker_id}\t{tasks_for_cur_worker}\t{sight.location.get()}\n')
      outf.write(f'{worker_id}\t{sight.location.get()}\n')
      sight.location.get().next()

  remote_script = binary_path

  script_args = (
      f'--decision_mode={decision_mode} --server_mode={server_mode} --worker_mode={worker_mode} --optimizer_type={optimizer_type}  --cache_mode={cache_mode} '
  )

  env_vars = [
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      f'PROJECT_ID={os.environ["PROJECT_ID"]}',
      '--env',
      f'GOOGLE_CLOUD_PROJECT={os.environ["PROJECT_ID"]}',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
      '--env',
      f'WORKERS_CONFIG_PATH={FLAGS.workers_config_path}',
      '--env',
      f'OPTIMIZERS_CONFIG_PATH={FLAGS.optimizers_config_path}',
  ]

  if FLAGS.server_mode == 'vm':
    if FLAGS.ip_addr == 'localhost':
      raise ValueError("ip_address must be provided for workers")
    env_vars += ['--env', f'IP_ADDR={FLAGS.ip_addr}']
  elif FLAGS.server_mode == 'local':
    env_vars += ['--env', f'IP_ADDR={service.get_docker0_ip()}']
  elif FLAGS.server_mode == 'cloud_run':
    env_vars += ['--env', f'SIGHT_SERVICE_ID={service._SERVICE_ID}']

  print('sight.id=%s' % sight.id)
  args = [
      'dsub',
      '--provider=local',
      f'--image={docker_image}',
      f'--project={_PROJECT_ID.value}',
      f'--logging=extra/dsub-logs',
      *env_vars,
      '--input',
      f'SCRIPT={remote_script}',
      '--input-recursive',
      f'CLOUDSDK_CONFIG={os.path.expanduser("~")}/.config/gcloud',
      f'--command=python3 "${{SCRIPT}}" {script_args}',
      '--tasks',
      '/tmp/optimization_tasks.tsv',
      '--name',
      _get_experiment_name(sight)[:63],
  ]
  logging.info('CLI=%s', ' '.join(args))
  subprocess.run(args, check=True)

  sight.exit_block('Worker Spawning', sight_pb2.Object())
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)
