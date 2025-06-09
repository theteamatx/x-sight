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

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from absl import flags
from absl import logging
from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_helper import CacheConfig
from helpers.cache.cache_helper import KeyMaker
from helpers.cache.cache_interface import CacheInterface
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import trials
from sight.widgets.decision import utils
from sight.widgets.decision.resource_lock import RWLockDictWrapper
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient
)

global_outcome_mapping = RWLockDictWrapper()

FLAGS = flags.FLAGS


async def push_message(sight_id, action_id):
  try:
    global_outcome_mapping.set_for_key(action_id, None)
  except Exception as e:
    print(f'Exception => {e}')
    raise e


async def fetch_outcome(sight_id, actions_id):
  while True:
    try:
      outcome = global_outcome_mapping.get_for_key(actions_id)
      if outcome:
        return outcome
      else:
        # async_dict = global_outcome_mapping.get()
        # print(f'GLOBAL_MAPPING_GET_OUTCOME_QUEUE => {async_dict}')
        time = 5
        # print(f'Waiting for {actions_id} for {time} seconds...')
        await asyncio.sleep(time)
    except Exception as e:
      raise e


async def asyncio_wrapper(blocking_func, *args, max_threads=-1):
  """Wrapper to execute a blocking function using asyncio.to_thread.

  Parameters:
      blocking_func (callable): The blocking function to execute.
      *args: Positional arguments to pass to the blocking function.
      max_threads (int): Number of threads for the custom ThreadPoolExecutor.
                         If -1, use the default executor.

  Returns:
      The result of the blocking function.
  """
  if max_threads != -1:
    # Create a custom ThreadPoolExecutor
    custom_executor = ThreadPoolExecutor(max_workers=max_threads,
                                         thread_name_prefix="CustomThread")
    try:
      # Temporarily set the custom executor
      loop = asyncio.get_running_loop()
      loop.set_default_executor(custom_executor)
      print(f"Using custom thread pool with max threads: {max_threads}")
      return await asyncio.to_thread(blocking_func, *args)
    finally:
      # Shutdown the custom executor after usage
      custom_executor.shutdown(wait=True)
  else:
    print("Using default thread pool")
    # Use the default executor
    return await asyncio.to_thread(blocking_func, *args)


async def propose_actions(sight,
                          question_label,
                          action_dict,
                          custom_part="sight_cache"):

  if (not global_outcome_mapping.get_for_key(
      f'is_poll_thread_started_{question_label}')):
    decision.init_sight_polling_thread(sight.id, question_label)
    global_outcome_mapping.set_for_key(
        f'is_poll_thread_started_{question_label}', True)

  key_maker = KeyMaker()
  worker_version = utils.get_worker_version(question_label)
  custom_part = custom_part + ':' + worker_version
  cache_key = key_maker.make_custom_key(custom_part, action_dict)

  cache_client = CacheFactory.get_cache(
      FLAGS.cache_mode,
      # * Update the config as per need , None config means it takes default redis config for localhost
      with_redis=CacheConfig.get_redis_instance(FLAGS.cache_mode, config=None))

  outcome = cache_client.get(key=cache_key)

  if outcome is not None:
    outcome = json.loads(outcome)
    print('Getting response from cache !!')
    return outcome

  # unique_action_id = decision.propose_actions(sight, question_label, action_dict)
  # unique_action_id = await asyncio.to_thread(decision.propose_actions, sight,
  #                                            action_dict)
  unique_action_id = await asyncio_wrapper(decision.propose_actions, sight,
                                           question_label, action_dict)
  await push_message(sight.id, unique_action_id)
  response = await fetch_outcome(sight.id, unique_action_id)
  outcome = response.get('outcome', None)
  if response is None or outcome is None:
    raise Exception('fetch_outcome response or respose["outcome"] is none')
  # converting the stringify data into json data if it can
  for key in outcome:
    value = outcome[key]
    try:
      final_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
      final_value = value
    outcome[key] = final_value
  logging.info('cache_key=%s', cache_key)
  logging.info('outcome=%s', outcome)
  cache_client.set(key=cache_key, value=json.dumps(outcome))
  return outcome
