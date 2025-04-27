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
"""Helper methods to handle server communication."""

import os
import random
import re
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

from absl import flags
from dotenv import load_dotenv
import google.auth.transport.requests
import google.oauth2.id_token
import grpc
from helpers.logs.logs_handler import logger as logging
import requests
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc

load_dotenv()
FLAGS = flags.FLAGS

current_script_directory = os.path.dirname(os.path.abspath(__file__))
#  Check which crt will works for you
# _CERT_FILE_PATH = "/etc/ssl/certs/ca-certificates.crt"
# _CERT_FILE_PATH = os.path.join(current_script_directory, '..', '..',
#                                'sight_service', 'sight_service.cert')
_SERVICE_NAME = flags.DEFINE_string(
    'service_name',
    '',
    'The name of the Sight service instance that will be managing this run.',
)
_SERVICE_DOCKER_FILE = flags.DEFINE_string(
    'service_docker_file',
    '',
    'path of docker file to be used while deploying service',
)
_SERVICE_DOCKER_IMG = flags.DEFINE_string(
    'service_docker_img',
    '',
    'name of local docker image to be used while deploying service',
)
_IP_ADDR = flags.DEFINE_string(
    'ip_addr',
    'localhost',
    'where the service is deployed',
)
_PORT = flags.DEFINE_string(
    'port',
    '443',
    'port number on which service is deployed',
)
_SERVER_MODE = flags.DEFINE_enum(
    'server_mode',
    None,
    ['vm', 'cloud_run', 'local'],
    ('The procedure to use when training a model to drive applications that '
     'use the Decision API.'),
)
_SIGHT_SERVICE_KNOWN = False
_SERVICE_ID = ''
_RESPONSE_TIMES = []
_UNIQUE_STRING = ''
_SERVICE_PREFIX = 'sight-'


def get_service_id() -> str:
  """find service id for sight server.

  Returns:
    service id
  """
  global _SERVICE_ID
  global _SIGHT_SERVICE_KNOWN

  # print('os.environ : ', os.environ)
  if 'SIGHT_SERVICE_ID' in os.environ:
    # print('used env flow from get_service_id.....')
    _SERVICE_ID = os.environ['SIGHT_SERVICE_ID']
  elif _SIGHT_SERVICE_KNOWN:
    return _SERVICE_ID
  elif _SERVICE_NAME.value:
    _SERVICE_ID = _SERVICE_NAME.value
  elif _SERVICE_DOCKER_FILE.value or _SERVICE_DOCKER_IMG.value:
    _SERVICE_ID = str(uuid.uuid4())
  else:
    _SERVICE_ID = 'default'
  _SIGHT_SERVICE_KNOWN = True

  # logging.info("service id : %s%s", _SERVICE_PREFIX, _SERVICE_ID)
  return _SERVICE_ID


def get_port_number() -> str:
  # logging.info('FLAGS.port is %s', FLAGS.port)
  # logging.info('_PORT.value is %s', _PORT.value)

  # logging.info(
  #     'in get_port_number => os.environ.PORT => %s FLAGS.server_mode => %s  ',
  #     os.environ.get('PORT', 'None'), FLAGS.server_mode)

  if (FLAGS.server_mode in ['local', 'vm']):
    return '8080'
  else:
    return _PORT.value


def _service_addr() -> str:
  # return f'{_SERVICE_PREFIX}{get_service_id()}-dq7fdwqgbq-uc.a.run.app'
  global _UNIQUE_STRING
  # if('IP_ADDR' in os.environ):
  #     return os.environ['IP_ADDR']
  # elif (_UNIQUE_STRING):
  if (_UNIQUE_STRING):
    # print("unique string found : ", _UNIQUE_STRING)
    # print("get_service_id() : ", get_service_id())
    return f'{_SERVICE_PREFIX}{get_service_id()}-{_UNIQUE_STRING}-uc.a.run.app'
  else:
    # print('fetching unique string.....')
    try:
      print('get_service_id()=', get_service_id())
      service_url = subprocess.getoutput(
          'gcloud run services describe'
          f" {_SERVICE_PREFIX}{get_service_id()} --region us-central1 --format='value(status.url)'"
      )
      print("service url : ", service_url)
      _UNIQUE_STRING = re.search(r'https://.*-(\w+)-uc\.a\.run\.app',
                                 service_url).group(1)
      print("_UNIQUE_STRING : ", _UNIQUE_STRING)
    except Exception as e:
      logging.exception("service not found : %s", e)
    # print("first _UNIQUE_STRING : ", _UNIQUE_STRING)
    return f'{_SERVICE_PREFIX}{get_service_id()}-{_UNIQUE_STRING}-uc.a.run.app'


def _find_or_deploy_server() -> str:
  """deploy sight server with given docker image."""

  global _SIGHT_SERVICE_KNOWN
  if (os.environ.get('SIGHT_SERVICE_ID')):
    # print('service found from environment variable : ', get_service_id())
    # logging.info('service found from environment variable')
    return get_service_id()

  if _SIGHT_SERVICE_KNOWN or (not _SERVICE_DOCKER_FILE.value and
                              not _SERVICE_DOCKER_IMG.value):
    try:
      # get list of services deployed on cloud-run which
      # includes given service-name
      response = subprocess.getoutput(
          'gcloud run services list'
          f" --filter='SERVICE:{_SERVICE_PREFIX}{get_service_id()}'")

      if response == 'Listed 0 items.':
        # given service_name doesn't exist on cloud-run
        if get_service_id() != 'default':
          raise ValueError(
              f"{_SERVICE_PREFIX}{get_service_id()} doesn't exist, try with"
              ' different name...')
        else:
          logging.info('No such service exist : %s',
                       _SERVICE_PREFIX + get_service_id())
          logging.info('creating new service : %s%s', _SERVICE_PREFIX,
                       get_service_id())

      else:
        # given service_name exist on cloud-run
        return get_service_id()
    except ValueError as e:
      logging.info('value Error : %s', e)
      sys.exit(0)
    except Exception:
      # error while calling "gcloud run service list" command
      try:
        # sample Test call is possible if service exist
        sight_service = obtain_secure_channel()
        metadata = []
        id_token = generate_id_token()
        metadata.append(('authorization', 'Bearer ' + id_token))
        # print("try in calling dummt test service call")
        response = sight_service.Test(service_pb2.TestRequest(),
                                      300,
                                      metadata=metadata)
        return get_service_id()
      except Exception as error:
        logging.info(
            "Provided service - %s doesn't exist or Not enough permissions: %s",
            get_service_id(),
            error,
        )
        sys.exit(0)

  # deploy new service
  print('_SERVICE_ID=', get_service_id())
  docker_file_path = _SERVICE_DOCKER_FILE.value
  docker_img = _SERVICE_DOCKER_IMG.value

  if get_service_id() == 'default' and not _SERVICE_DOCKER_FILE.value:
    docker_file_path = 'service/Dockerfile'
    # elif(not _SERVICE_DOCKER_FILE.value):
    # raise ValueError(
    #     'flag --service_docker_file required with any new service-name'
    # )

  if (docker_file_path):
    logging.info('building img from scratch.....................')
    # Step 1: Build docker image
    build_out = subprocess.run(
        [
            'docker',
            'build',
            '-t',
            f'gcr.io/{os.environ["PROJECT_ID"]}/{_SERVICE_PREFIX}' +
            get_service_id(),
            '-f',
            docker_file_path,
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
        [
            'docker', 'push',
            f'gcr.io/{os.environ["PROJECT_ID"]}/{_SERVICE_PREFIX}' +
            get_service_id()
        ],
        check=True,
        capture_output=True,
    )
    # logging.info('push_out=%s', push_out)

    # Step 5: fetch image id
    image_id = ''
    for line in push_out.stdout.decode('utf-8').splitlines():
      m = re.search(r'sha256:([a-z0-9]+) size: [0-9]+$', line)
      if m:
        image_id = m.group(1)
    if not image_id:
      raise ValueError(
          f'Failed to find image id in output of docker push:\n{push_out.stdout}'
      )
    logging.info('image_id=%s', image_id)

  if (docker_file_path):
    logging.info('using newly build img to deploy service')
    img_name = f'gcr.io/{os.environ["PROJECT_ID"]}/{_SERVICE_PREFIX}{get_service_id()}@sha256:{image_id}'
  elif (docker_img):
    logging.info('using docker img to deploy service')
    img_name = _SERVICE_DOCKER_IMG.value
  else:
    raise ValueError(
        'img_name have to specify before deplying cloud run service')

  # Step 6: deploy cloud run service from deployed image
  deploy_out = subprocess.run(
      [
          'gcloud',
          'run',
          'deploy',
          _SERVICE_PREFIX + get_service_id(),
          f'--image={img_name}',
          '--allow-unauthenticated',
          f'--service-account={flags.FLAGS.service_account}@{os.environ["PROJECT_ID"]}.iam.gserviceaccount.com',
          '--concurrency=default',
          '--cpu=4',
          '--memory=16Gi',
          '--min-instances=1',
          '--max-instances=1',
          '--no-cpu-throttling',
          '--region=us-central1',
          f'--project={os.environ["PROJECT_ID"]}',
      ],
      check=True,
      capture_output=True,
  )
  print('deploy_out : ', deploy_out.stderr)

  logging.info('_SERVICE_ID=%s', _SERVICE_ID)
  if (docker_file_path):
    logging.info('deleting newly built img')
    subprocess.run(
        [
            'gcloud',
            'container',
            'images',
            'delete',
            f'gcr.io/{os.environ["PROJECT_ID"]}/{_SERVICE_PREFIX}{get_service_id()}@sha256:{image_id}',
            '--quiet',
            '--force-delete-tags',
        ],
        check=True,
    )
  # logging.info('%s', ' '.join(['gcloud', 'run', 'services', 'delete',
  #                 get_service_id(),
  #                '--region=us-central1', '--quiet']))
  # subprocess.run(['gcloud', 'run', 'services', 'delete',
  #                 _SIGHT_SERVICE_ADDR,
  #                '--region=us-central1', '--quiet'],
  #                check=True)

  # _SIGHT_SERVICE_ADDR=f'{service_id}-dq7fdwqgbq-uc.a.run.app'
  _SIGHT_SERVICE_KNOWN = True

  print(
      'Log:'
      f' https://pantheon.corp.google.com/run/detail/us-central1/{_SERVICE_PREFIX}{get_service_id()}/logs?project={os.environ["PROJECT_ID"]}'
  )

  return get_service_id()


def finalize_server() -> None:
  # if _SERVICE_DOCKER_FILE.value:
  #   subprocess.run(['gcloud', 'run', 'services', 'delete',
  #                   get_service_id(),
  #                 '--region=us-central1', '--quiet'],
  #                 check=True)
  pass


def get_id_token_of_service_account(user_access_token, service_account,
                                    url) -> str:
  """fetch id_token for given service_account using user credentials.

  Args:
    user_access_token: token to verify identity of user generating credentials
      of service_account
    service_account: account_name for which id_token is requested
    url: url for which this token will be valid

  Returns:
    id_token: id_token of service_account
  """
  headers = {
      'Authorization': 'Bearer ' + user_access_token,
      'Content-Type': 'application/json; charset=utf-8',
  }
  data = b'{"audience": "%s"}' % url.encode('utf-8')

  try:
    response = requests.post(
        'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/' +
        service_account + ':generateIdToken',
        headers=headers,
        data=data,
    )
    return response.json()['token']
  except Exception as e:
    logging.info('API CALL ERROR: %s', e)


def generate_id_token():
  """fetch id_token for given user.

  Returns:
    id_token: id_token of user_account
  """
  # fetch id-token of service account from which we spawned D-SUB
  # worker in cloud

  if 'worker_mode' in flags.FLAGS and flags.FLAGS.worker_mode == 'dsub_cloud_worker':
    logging.info('using credentials of service account for : https://%s',
                 _service_addr())
    auth_req = google.auth.transport.requests.Request()
    service_account_id_token = google.oauth2.id_token.fetch_id_token(
        auth_req, 'https://' + _service_addr())
    id_token = service_account_id_token
  # fetch id-token locally
  else:
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    # impersonating service-account if passed in parameter

    if 'service_account' in flags.FLAGS and flags.FLAGS.service_account != None:
      # print("using service account's credentils..... :")
      user_access_token = creds.token
      service_account = f'{flags.FLAGS.service_account}@{os.environ["PROJECT_ID"]}.iam.gserviceaccount.com'
      url = f'https://{_service_addr()}'
      service_account_id_token = get_id_token_of_service_account(
          user_access_token, service_account, url)
      id_token = service_account_id_token
    # using user credentials
    else:
      user_id_token = creds.id_token
      id_token = user_id_token

  return id_token


def get_docker0_ip():
  try:
    # Run the 'ip addr' command
    output = subprocess.check_output(['ip', 'addr'], encoding='utf-8')

    # Regex to find the IP address for the docker0 interface
    match = re.search(r'docker0.*?inet (\d+\.\d+\.\d+\.\d+)', output, re.DOTALL)
    if match:
      return match.group(1)
    else:
      raise RuntimeError("docker0 interface not found")
  except Exception as e:
    return f"Error: {e}"


def obtain_secure_channel(options=None):
  """create secure channel to communicate with server.

  Returns:
    service_handle: to communicate with server
  """
  # hosted server
  # if 'SIGHT_SERVICE_PATH' in os.environ:
  #   cert_file = (f'{os.environ["SIGHT_SERVICE_PATH"]}/sight_service.cert')
  # else:
  #   cert_file = _CERT_FILE_PATH
  # with open(cert_file, 'rb') as f:

  creds = grpc.ssl_channel_credentials()

  # if('IP_ADDR' in os.environ):
  #   url = os.environ['IP_ADDR']
  # else:
  url = _service_addr()
  target = '{}:{}'.format(url, get_port_number())
  logging.info("secure channel : target %s , creds %s and options %s here ", target, creds,
               options)
  channel = grpc.secure_channel(
      target,
      creds,
      options,
  )
  return channel


def obtain_insecure_channel(options):
  """create insecure channel to communicate with server.

  Returns:
    service_handle: to communicate with server
  """
  # server_mode is VM or (local in dsub worker)
  if 'IP_ADDR' in os.environ:
    host = os.environ["IP_ADDR"]
  # elif FLAGS.worker_mode=='dsub_local_worker':
  #   host = get_docker0_ip()
  # server_mode is local in client
  else:
    host = 'localhost'
  target = '{}:{}'.format(host, get_port_number())
  # print("service_url here : ", target)

  logging.info("Insecure channel : target %s , and options %s here ", target, options)

  channel = grpc.insecure_channel(
      target,
      options,
  )
  return channel


class GRPCClientCache:
  _secure_cache = None
  _insecure_cache = None

  @classmethod
  def generate_metadata(cls):
    """Generate metadata to call service with authentication."""

    logging.debug('_secure_cache %s and _insecure_cache %s ', cls._secure_cache,
                 cls._insecure_cache)

    channel_opts = [
        ('grpc.max_send_message_length', 512 * 1024 * 1024),
        ('grpc.max_receive_message_length', 512 * 1024 * 1024),
    ]

    if FLAGS.server_mode in ['local', 'vm']:
      if cls._insecure_cache is None:
        channel = obtain_insecure_channel(channel_opts)
        sight_service = service_pb2_grpc.SightServiceStub(channel)
        metadata = []
        cls._insecure_cache = (sight_service, metadata)
      return cls._insecure_cache

    else:
      if cls._secure_cache is None:
        # for client code, need to find or deploy cloud run service, workers will directly get via env
        if FLAGS.worker_mode is None:
          _find_or_deploy_server()
        secure_channel = obtain_secure_channel()
        # print("secure_channel : ", secure_channel)
        sight_service = service_pb2_grpc.SightServiceStub(secure_channel)
        metadata = []
        id_token = generate_id_token()
        # print('id_token : ', id_token)
        metadata.append(('authorization', 'Bearer ' + id_token))
        cls._secure_cache = (sight_service, metadata)
      return cls._secure_cache


# def calculate_response_time(start_time):
#   response_time = time.time() - start_time
#   print(f'Response Time From Server Inside Service call: {round(response_time,4)} seconds')
# _RESPONSE_TIMES.append(response_time)
# avg_response_time = sum(_RESPONSE_TIMES) / len(_RESPONSE_TIMES)
# print(f'Average Response Time From Server Inside Service call: {round(avg_response_time,4)} seconds')
# data_structures.log(round(response_time, 4), self._sight)


def call(invoke_func: Callable[[Any, Any], Any]) -> Any:
  """Calls invoke_func as many times as needed for it to complete.

  After each failed call (RPCError raised), this function backs off for a
  random exponentially increasing time period and retries.

  Args:
    invoke_func: method_name at server to be called

  Returns:
    response: response received from server side after invoking the function
  """
  sight_service, metadata = GRPCClientCache.generate_metadata()
  num_retries = 0
  backoff_interval = 0.5
  while True:
    try:
      response = invoke_func(sight_service, metadata)
      return response
    except grpc.RpcError as e:
      logging.info('RPC ERROR: %s', e)
      if e.code() == grpc.StatusCode.PERMISSION_DENIED:
        print('NO ACCESS!!!!')
      elif e.code() == grpc.StatusCode.UNIMPLEMENTED:
        print('SIGHT SERVICE NOT FOUND!!!')
      if num_retries == 12:
        raise e
      time.sleep(random.uniform(backoff_interval / 2, backoff_interval))
      logging.info('backed off for %s seconds...', backoff_interval)
      backoff_interval *= 2
      num_retries += 1
