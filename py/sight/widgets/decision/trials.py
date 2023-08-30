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
import subprocess
from typing import Any, Dict, Optional

from sight.proto import sight_pb2
from absl import flags
from absl import logging
from sight.service import generate_metadata
from service.decision import decision_pb2

_VIZIER_TRAIN_EXPERIMENT_NAME = flags.DEFINE_string(
    'vizier_train_experiment_name', None,
    'The name of the Vizier experiment this worker will participate in.')
_NUM_TRIALS = flags.DEFINE_integer(
    'vizier_num_trials', 5, 'Number of Vizier trials to perform.')
_PROJECT_ID = flags.DEFINE_string(
    'project_id', 'cameltrain', 'Id of project')
_PROJECT_REGION = flags.DEFINE_string(
    'region', 'us-central1', 'location to store study')

_DSUB_PROVIER = flags.DEFINE_string(
    'dsub_provider', 'google-cls-v2', '')
_DSUB_REGION = flags.DEFINE_string(
    'dsub_region', 'us-west1', '')
_DSUB_IMAGE = flags.DEFINE_string(
    'dsub_image', 'gcr.io/cameltrain/sight-worker', '')
_DSUB_MACHINE_TYPE = flags.DEFINE_string(
    'dsub_machine-type', 'e2-standard-2', '')
_DSUB_LOGGING = flags.DEFINE_string(
    'dsub_logging', 'gs://dsub_cameltrain/sight/worker-sight-logging', '')
_DSUB_SERVICE_ACCOUNT = flags.DEFINE_string(
    'dsub_service-account', 'vizier-service-account@cameltrain.iam.gserviceaccount.com', '')
_DSUB_BOOT_DISK_SIZE = flags.DEFINE_integer(
    'dsub_boot-disk-size', 30, '')

_file_name = 'trials.py'

def _get_vizier_study_display_name(sight:Any) -> str:
  if _VIZIER_TRAIN_EXPERIMENT_NAME.value:
    return _VIZIER_TRAIN_EXPERIMENT_NAME.value
  else:
    return 'Sight_Decision_Study_' + sight.params.label.replace(' ', '_') + '_' + str(
        sight.id) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')

def launch(
    num_train_workers: int,
    binary_path: Optional[str],
    optimizer_type: str,
    sight: Any,
):
  """Launches the dsub workers that will run the optimization.

  Args:
    num_train_workers: Number of workers to use in a training run.
    binary_path: Path of the Blaze target of this binary.
    optimizer_type: Type of optimizer we are using.
    sight: The Sight object to be used for logging.
  """
  method_name = "launch"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  print("**********************************************************************")
  sight_service,metadata = generate_metadata()

  req = decision_pb2.LaunchRequest()

  config_param = sight_pb2.DecisionConfigurationStart()
  for action_attr in sight.widget_decision_state['decision_episode_fn'].action_attrs:
    AttrProps = sight_pb2.DecisionConfigurationStart.AttrProps()
    AttrProps.min_value = sight.widget_decision_state['decision_episode_fn'].action_min[action_attr]
    AttrProps.max_value = sight.widget_decision_state['decision_episode_fn'].action_max[action_attr]
    config_param.action_attrs[action_attr].CopyFrom(AttrProps)

  for state_attr in sight.widget_decision_state['decision_episode_fn'].state_attrs:
    AttrProps = sight_pb2.DecisionConfigurationStart.AttrProps()
    AttrProps.min_value = sight.widget_decision_state['decision_episode_fn'].state_min[state_attr]
    AttrProps.max_value = sight.widget_decision_state['decision_episode_fn'].state_max[state_attr]
    config_param.state_attrs[state_attr].CopyFrom(AttrProps)
  req.decision_config_params.CopyFrom(config_param)

  if(optimizer_type == "vizier"):
    req.optimizer_type = decision_pb2.OptimizerType.OT_VIZIER
    req.label=sight.params.label
    req.client_id = str(sight.id)
  elif(optimizer_type == "dm_acme"):
    req.optimizer_type = decision_pb2.OptimizerType.OT_ACME
  else:
    req.optimizer_type = decision_pb2.OptimizerType.OT_UNKNOWN

  try:
    # print("request for launch method is :", req)
    response = sight_service.Launch(req,  300, metadata=metadata)
    logging.info('##### response=%s #####', response)
  except Exception as e:
    logging.info('RPC ERROR: %s', e)  

  print("**********************************************************************")

  sight.enter_block("Worker Spawning", sight_pb2.Object())
  with open('/tmp/optimization_tasks.tsv', 'w') as outf:
    outf.write('--env worker_id\t--env num_samples\t--env worker_location\n')
    num_tasks_per_worker = math.floor(_NUM_TRIALS.value/num_train_workers)
    for worker_id in range(num_train_workers):
      tasks_for_cur_worker = num_tasks_per_worker
      # If _NUM_TRIALS is not evenly divisible by num_train_workers, add 
      # the extra extra tasks to the first few workers.
      if worker_id<_NUM_TRIALS.value % num_train_workers:
        tasks_for_cur_worker += 1
      outf.write(f'{worker_id}\t{tasks_for_cur_worker}\t{sight.location}\n')
      sight.location.next()

  REMOTE_SCRIPT=f'gs://dsub_cameltrain/cameltrain/' + binary_path.split('/')[-1]
  print(f'Uploading {binary_path}...')
  subprocess.run(['gsutil', 'cp', '-c', binary_path, REMOTE_SCRIPT])

  args = ['dsub', 
          f'--provider={_DSUB_PROVIER.value}',
          f'--regions={_DSUB_REGION.value}',
          f'--image={_DSUB_IMAGE.value}',
          f'--machine-type={_DSUB_MACHINE_TYPE.value}',
          f'--project={_PROJECT_ID.value}',
          f'--logging={_DSUB_LOGGING.value}',
          '--env', f'PARENT_LOG_ID={sight.id}',
          '--env', f'PYTHONPATH=/project', 
          '--input', f'SCRIPT={REMOTE_SCRIPT}',
          '--command=cd /project && python3 /mnt/data/input/gs/dsub_cameltrain/cameltrain/'+binary_path.split('/')[-1]+'  --decision_mode=train --decision_train_alg=worker_mode',
          f'--service-account={_DSUB_SERVICE_ACCOUNT.value}',
          f'--boot-disk-size={_DSUB_BOOT_DISK_SIZE.value}',
          '--tasks', '/tmp/optimization_tasks.tsv',
          '--name', _get_vizier_study_display_name(sight)]
  logging.info('CLI=%s', ' '.join(args))
  subprocess.run(args, check=True)

  print('check the status at : ', response.display_string)
  sight.exit_block('Worker Spawning', sight_pb2.Object())
  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")