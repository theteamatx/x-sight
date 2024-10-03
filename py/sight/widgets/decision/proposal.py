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
import json

from absl import flags
from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_helper import CacheConfig
from helpers.cache.cache_helper import CacheKeyMaker
from helpers.cache.cache_interface import CacheInterface
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import trials
from sight.widgets.decision.resource_lock import RWLockDictWrapper
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient
)

_CACHE_MODE = flags.DEFINE_enum(
    'cache_mode', 'none',
    ['gcs', 'local', 'redis', 'none', 'gcs_with_redis', 'local_with_redis'],
    'Which Sight cache to use ? (default is none)')

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


async def propose_actions(sight, action_dict, custom_part="sight_cache"):

  key_maker = CacheKeyMaker()
  cache_key = key_maker.make_custom_key(custom_part, action_dict)

  cache_client = CacheFactory.get_cache(
      _CACHE_MODE.value,
      with_redis=CacheConfig.get_redis_instance(_CACHE_MODE.value))

  outcome = cache_client.json_get(key=cache_key)

  if outcome is not None:
    print('Getting response from cache !!')
    return outcome

  unique_action_id = decision.propose_actions(sight, action_dict)
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
  cache_client.json_set(key=cache_key, value=outcome)
  return outcome
