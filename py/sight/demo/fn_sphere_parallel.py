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
"""Demo of using the Sight Decision API to run forest simulator."""
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import inspect
import multiprocessing
import os
import re
import subprocess
import time
from typing import Sequence

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight import service_utils as service
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight_service.proto import service_pb2

FLAGS = flags.FLAGS

_PROJECT_ID = 'cameltrain'
_DOCKER_FILE_PATH = 'sight_service/Dockerfile'
_SERVICE_PREFIX = 'sight-'


def build_push_service_img(sight_id):
  build_out = subprocess.run(
      [
          'docker',
          'build',
          '-t',
          f'gcr.io/{_PROJECT_ID}/{_SERVICE_PREFIX}' + sight_id,
          '-f',
          _DOCKER_FILE_PATH,
          '.',
      ],
      check=True,
      capture_output=True,
  )
  # logging.info('build_out=%s', build_out)

  # Step 2: Retrieve an OAuth2 access token
  access_token_cmd = ['gcloud', 'auth', 'print-access-token']
  access_token_result = subprocess.run(access_token_cmd,
                                       capture_output=True,
                                       text=True,
                                       check=True)
  access_token = access_token_result.stdout.strip()

  # Step 3: Authenticate with gcr.io using the access token
  login_cmd = [
      'docker',
      'login',
      '-u',
      'oauth2accesstoken',
      '-p',
      access_token,
      'https://gcr.io',
  ]
  subprocess.run(login_cmd, check=True)

  # Step 4: push created image to gcr.io
  push_out = subprocess.run(
      ['docker', 'push', f'gcr.io/{_PROJECT_ID}/{_SERVICE_PREFIX}' + sight_id],
      check=True,
      capture_output=True,
  )
  # logging.info('push_out=%s', push_out)

  return f'gcr.io/{_PROJECT_ID}/{_SERVICE_PREFIX}' + sight_id


def delete_service_img(sight_id):
  print('deleting image : gcr.io/' + _PROJECT_ID + '/' + _SERVICE_PREFIX +
        sight_id)
  subprocess.run(
      [
          'gcloud',
          'container',
          'images',
          'delete',
          f'gcr.io/{_PROJECT_ID}/{_SERVICE_PREFIX}{sight_id}',
          '--quiet',
          '--force-delete-tags',
      ],
      check=True,
  )


def delete_service(service_name):
  print('deleting sight service')
  sight_service_name = _SERVICE_PREFIX + service_name
  cmd_args = [
      'gcloud', 'run', 'services', 'delete', sight_service_name, '--quiet'
  ]
  result = subprocess.run(args=cmd_args, capture_output=True, text=True)
  # print('result from deletion :', result)
  if result.returncode == 0:
    print(
        f'Successfully deleted Cloud Run service: {_SERVICE_PREFIX}{service_name}'
    )
  else:
    print(f'Error deleting Cloud Run service: {result.stderr}')


def run_experiment(sight_id, optimizer_value, image_id, table_queue):
  cmd_args = [
      'python',
      'py/sight/demo/fn_sphere.py',
      '--decision_mode',
      'train',
      '--deployment_mode',
      'distributed',
      '--num_train_workers',
      '1',
      '--num_trials',
      '10',
      '--optimizer_type',
      optimizer_value,
      '--docker_image',
      'gcr.io/cameltrain/sight-worker',
      # '--service_docker_file', 'sight_service/Dockerfile'
      '--service_docker_img',
      image_id,
      '--parent_id',
      sight_id
  ]
  result = subprocess.run(args=cmd_args, capture_output=True, text=True)
  # print('here result is  : ', result.stdout)
  table_name = re.search(r'table generated\s*:\s*([^\s]+)', result.stdout)
  service_name = re.search(r'_SERVICE_ID=\s*([0-9a-fA-F-]+)', result.stdout)
  print('service_name : ', service_name.group(1))
  # raise SystemExit

  if (table_name and service_name):
    # print(result.stdout)
    table_queue.put(
        (optimizer_value, table_name.group(1), service_name.group(1)))
  # else:
  print(f'whole log from {optimizer_value} : ', result.stderr)


def get_sight_instance():
  params = sight_pb2.Params(
      label='sphere_parallel',
      bucket_name=f'{_PROJECT_ID}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def check_exp_status(exp_sight_id, exp_service_id):
  print('in check exp_status .........................')
  print(exp_sight_id, exp_service_id)
  os.environ['SIGHT_SERVICE_ID'] = exp_service_id
  req = service_pb2.CurrentStatusRequest()
  req.client_id = exp_sight_id
  response = service.call(
      lambda s, meta: s.CurrentStatus(req, 300, metadata=meta))
  print('response :', response.status)
  if (response.status == service_pb2.CurrentStatusResponse.Status.SUCCESS):
    return True
  else:
    return False


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    image_id = build_push_service_img(str(sight.id))
    print('image_id : ', image_id)
    sight.text(image_id)

    # optimizer_values = ['ng_random_search', 'ng_pso', 'ng_cga']
    # optimizer_values = ['ng_random_search', 'ng_pso', 'ng_cga', 'ng_es', 'ng_dl_opo', 'ng_dde']
    optimizer_values = ['ng_cga']
    # optimizer_values = ['bayesian_opt']

    # optimizer_values = [
    #     'ng_random_search', 'ng_pso', 'ng_de', 'ng_cga', 'ng_es', 'ng_dl_opo', 'ng_dde',
    #     'ng_nmm', 'ng_tiny_spsa', 'ng_scr_hammersley_search',
    #     'ng_two_points_de', 'ng_cma_small', 'ng_cma', 'ng_auto', 'ng_bo',
    #     'ng_voronoi_de', 'bayesian_opt'
    # ]
    table_queue = multiprocessing.Queue()
    processes = []

    for optimizer_value in optimizer_values:
      process = multiprocessing.Process(target=run_experiment,
                                        args=(str(sight.id), optimizer_value,
                                              image_id, table_queue))
      processes.append(process)
      process.start()
    print('all process started.....')

    for process in processes:
      process.join()
    print('all process finished.....')

    delete_service_img(str(sight.id))

    experiment_details = {}
    while not table_queue.empty():
      optimizer_value, table_name, service_name = table_queue.get()
      with Block("Superscript Experiment Details", sight):
        with Attribute("optimizer", optimizer_value, sight):
          sight_id_match = re.search(r'\.(.*?)_log$', table_name)
          exp_sight_id = sight_id_match.group(1)
          # with Attribute("sight_id", exp_sight_id, sight):
          #   with Attribute("table_name", table_name, sight):
          # sight.text(f"{optimizer_value}:{exp_sight_id}")
          sight_obj = sight_pb2.Object()
          sight_obj.sub_type = sight_pb2.Object.SubType.ST_LINK
          sight_obj.link.linked_sight_id = str(exp_sight_id)
          sight_obj.link.link_type = sight_pb2.Link.LinkType.LT_PARENT_TO_CHILD
          frame = inspect.currentframe().f_back.f_back.f_back
          sight.set_object_code_loc(sight_obj, frame)
          sight.log_object(sight_obj, True)
          experiment_details[optimizer_value] = [
              exp_sight_id, table_name, service_name
          ]

    print('experiment_details : ', experiment_details)

    print('waiting for all experiments to get completed.......')
    completed_services = []
    while True:
      print(
          "checking if remaining experiments got compelted or not, to delete it's service"
      )
      # completed_services = []
      # for k,v in experiment_details.items():
      #     if check_exp_status(v[0], v[2]):
      #         service_name = v[2]
      #         sight_id = v[0]
      #         completed_services.append(service_name)
      #         del experiment_details[k]
      #         delete_service(service_name)

      for k in list(experiment_details.keys()):
        v = experiment_details[k]
        if check_exp_status(v[0], v[2]):
          service_name = v[2]
          # sight_id = v[0]
          completed_services.append(service_name)
          del experiment_details[k]
          delete_service(service_name)

      # Check if all services have succeeded
      print('completed_services : ', completed_services)
      if len(completed_services) == len(optimizer_values):
        # print()
        break  # All services have succeeded, exit loop

      # Wait for some time before polling again
      print('going in sleep mode for 60 sec')
      time.sleep(60)  # Polling interval of 60 seconds

    logging.info(
        'Log GUI : https://streamlit-app-dq7fdwqgbq-uc.a.run.app/?'
        'log_id=%s', str(sight.id))


if __name__ == "__main__":
  app.run(main)
