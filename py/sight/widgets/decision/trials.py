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
from absl import logging
import grpc
from sight_service.proto import service_pb2

from dotenv import load_dotenv
from sight import service_utils as service

from sight.proto import sight_pb2
from sight.widgets.decision.acme import acme_optimizer_client
from sight.widgets.decision.optimizer_client import OptimizerClient

load_dotenv()

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'The name of the experiment this worker will participate in.',
)
# _PROJECT_ID = flags.DEFINE_string(
#     'project_id', None, 'Id of cloud project'
# )
_PROJECT_ID = flags.DEFINE_string(
    'project_id', os.environ['PROJECT_ID'], 'Id of cloud project'
)
_PROJECT_REGION = flags.DEFINE_string(
    'project_region', 'us-central1', 'location to store project-data'
)
_DSUB_MACHINE_TYPE = flags.DEFINE_string(
    'dsub_machine-type', 'e2-standard-2', ''
)
# _DSUB_LOGGING = flags.DEFINE_string(
#     'log_path',
#     # 'tmp/logs',
#     f'gs://{os.environ["PROJECT_ID"]}/d-sub/logs/default',
#     'storage URI to store d-sub logs',
# )

# _DSUB_LOCAL_LOGGING = flags.DEFINE_string(
#     'dsub_local_logging',
#     None,#'logs/Dsub_local_logs',
#     'file path to store d-sub logs'
# )

_DSUB_BOOT_DISK_SIZE = flags.DEFINE_integer('dsub_boot_disk_size', 30, '')

_file_name = 'trials.py'
FLAGS = flags.FLAGS


def _get_experiment_name(sight: Any) -> str:
  if _EXPERIMENT_NAME.value:
    return _EXPERIMENT_NAME.value
  else:
    return (
        'Sight_Decision_Study_'
        + sight.params.label.replace(' ', '_')
        + '_'
        + str(sight.id)
        + '_'
        + datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def launch(
    # optimizer_type: str,
    # optimizer_config: Any,
    # state_attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps],
    # action_attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps],
    optimizer: OptimizerClient,
    decision_configuration: sight_pb2.DecisionConfigurationStart,
    num_train_workers: int,
    sight: Any,
):
  """Launches the experiment with the service.

  Args:
    optimizer_type: Type of optimizer we are using.
    state_attrs: maps the name of each state variable to its possible values.
    action_attrs: maps the name of each variable that describes possible
      decisions to its possible values.
    num_train_workers: numbers of workers to be spawned
    sight: The Sight object to be used for logging.
  """
  method_name = 'launch'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)
  logging.info('decision_configuration=%s' % decision_configuration)

  req = service_pb2.LaunchRequest()

  # config_param = sight_pb2.DecisionConfigurationStart()
  # for key, attr in action_attrs.items():
  #   config_param.action_attrs[key].CopyFrom(attr)
  # for key, attr in state_attrs.items():
  #   config_param.state_attrs[key].CopyFrom(attr)
  req.decision_config_params.CopyFrom(decision_configuration)

  req.label = sight.params.label
  req.client_id = str(sight.id)

  response = service.call(lambda s, meta: s.Launch(req, 300, metadata=meta))
  logging.info('##### Launch response=%s #####', response)

  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def start_job_in_docker(
    num_trials: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image: str,
    decision_mode: str,
    deployment_mode: str,
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
    deployment_mode: add
    worker_mode: add
    decision_params: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_job_in_docker'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  sight.enter_block('Worker Spawning', sight_pb2.Object())
  # Write the script that will execute the binary within the docker container.
  decision_params_arg = (
      f' --decision_params={decision_params}' if decision_params else ''
  )
  os.makedirs('/tmp/sight_script', exist_ok=True)
  with open('/tmp/sight_script/sight_decision_command.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write(
        '/usr/bin/python3'
        f' /project/{binary_path.split("/")[-1]} --decision_mode={decision_mode} --deployment_mode={deployment_mode}'
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
      # f'{os.path.expanduser("~")}/.config/gcloud:/project/.config/gcloud:ro',
      f'{FLAGS.gcloud_dir_path}:/project/.config/gcloud:ro',
      '--env',
      'GOOGLE_APPLICATION_CREDENTIALS=/project/.config/gcloud/application_default_credentials.json',
      '--env',
      'PYTHONPATH=/project',
      '--env',
      f'GOOGLE_CLOUD_PROJECT={_PROJECT_ID.value}',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
      # '--env',
      # f'SIGHT_SERVICE_ACCOUNT={_SERVICE_ACCOUNT.value}',
      '--env',
      f'worker_location={sight.location}',
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


def start_jobs(
    num_train_workers: int,
    num_trials: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image,
    decision_mode: str,
    deployment_mode: str,
    worker_mode: str,
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
    deployment_mode: add
    worker_mode: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_jobs'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

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

  # provider = 'local' if deployment_mode == 'local' else 'google-cls-v2'

  command = (
      'cd /x-sight && python3 "${SCRIPT}"'
      + f' --decision_mode={decision_mode}'
      + f' --deployment_mode={deployment_mode}'
      + f' --worker_mode={worker_mode}'
      + f' --optimizer_type={optimizer_type}'
      # + f' --project_id={os.environ["PROJECT_ID"]}'
  )
  if FLAGS.env_name:
    command += f' --env_name={FLAGS.env_name}'

  print('sight.id=%s' % sight.id)
  args = [
      'dsub',
      '--provider=google-cls-v2',
      f'--regions={_PROJECT_REGION.value}',
      f'--image={docker_image}',
      f'--machine-type={_DSUB_MACHINE_TYPE.value}',
      f'--project={_PROJECT_ID.value}',
      f'--logging=gs://{os.environ["PROJECT_ID"]}-sight/d-sub/logs/{service._SERVICE_ID}/{sight.id}',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      # '--env',
      # 'PYTHONPATH=/project',
      '--env',
      f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
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
  logging.info('worker logs available at : %s', f'gs://{os.environ["PROJECT_ID"]}/d-sub/logs/default')
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)


def start_job_in_dsub_local(
    num_train_workers: int,
    num_trials: int,
    binary_path: Optional[str],
    optimizer_type: str,
    docker_image,
    decision_mode: str,
    deployment_mode: str,
    worker_mode: str,
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
    deployment_mode: add
    worker_mode: add
    sight: The Sight object to be used for logging.
  """
  method_name = 'start_job_in_dsub_local'
  logging.debug('>>>>>>>>>  In %s method of %s file.', method_name, _file_name)

  sight.enter_block('Worker Spawning locally', sight_pb2.Object())
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
      f'gs://{os.environ["PROJECT_ID"]}/sight/d-sub/binary/' + binary_path.split('/')[-1]
  )
  print(f'Uploading {binary_path}...')
  subprocess.run(['gsutil', 'cp', '-c', binary_path, remote_script], check=True)

  # provider = 'google-cls-v2' if deployment_mode == 'distributed' else 'local'

  script_args = (
      f'--decision_mode={decision_mode} --deployment_mode={deployment_mode} --worker_mode={worker_mode} --optimizer_type={optimizer_type} '
  )
  if FLAGS.service_account:
    script_args = (
        script_args + f'--service_account={FLAGS.service_account}'
    )

  print('sight.id=%s' % sight.id)
  args = [
      'dsub',
      '--provider=local',
      f'--image={docker_image}',
      f'--project={_PROJECT_ID.value}',
      f'--logging=gs://{os.environ["PROJECT_ID"]}/d-sub/logs/local/{sight.id}',
      '--env',
      f'GOOGLE_CLOUD_PROJECT={os.environ["PROJECT_ID"]}',
      '--env',
      'GOOGLE_APPLICATION_CREDENTIALS=/mnt/data/mount/file'
      + f'{FLAGS.gcloud_dir_path}/application_default_credentials.json',
      '--env',
      f'PARENT_LOG_ID={sight.id}',
      '--env',
      'PYTHONPATH=/project',
      '--env',
      f'SIGHT_SERVICE_ID={service._SERVICE_ID}',
      '--input',
      f'SCRIPT={remote_script}',
      f'--command=cd /x-sight && python3 "${{SCRIPT}}" {script_args}',
      # + f'--optimizer_type={optimizer_type}',
      '--mount',
      'RESOURCES=file:/' + f'{FLAGS.gcloud_dir_path}',
      # + f'{os.path.expanduser("~")}/.config/gcloud',
      '--tasks',
      '/tmp/optimization_tasks.tsv',
      '--name',
      _get_experiment_name(sight)[:63],
  ]
  logging.info('CLI=%s', ' '.join(args))
  subprocess.run(args, check=True)

  sight.exit_block('Worker Spawning', sight_pb2.Object())
  logging.debug('<<<<<<<<<  Out %s method of %s file.', method_name, _file_name)
