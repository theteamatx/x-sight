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
"""normalize action attributes to required ranges."""

import copy


def scale_value(value, output_min, output_max, input_min=0, input_max=1):

  # Check if input value is within the source range
  if not (input_min <= value <= input_max):
    raise ValueError("Input value is outside the input range")

  # Calculate the normalized value within the input range
  normalized_value = (value - input_min) / (input_max - input_min)
  # print('normalized_value : ', normalized_value)

  # Scale the normalized value to the output range
  scaled_value = normalized_value * (output_max - output_min) + output_min
  # print('scaled_value : ', scaled_value)

  return scaled_value


class Normalizer:
  """provide interface to normalize - denormalize the action values in specified ranges.
    """

  def __init__(self):
    self.actions = {}
    self.normalized_actions = {}
    self.denormalized_actions = {}

  def normalize_in_0_to_1(self, actions):
    # storing key-value pair of action name and attr_proto in self.actions
    self.actions = copy.deepcopy(actions)

    # normalizing min max value of actions to 0-1
    for k, v in actions.items():
      current_action = actions[k]
      current_action.min_value = 0
      current_action.max_value = 1
      self.normalized_actions[k] = current_action

    # return key-value pair of action name and normalized attr_proto
    return self.normalized_actions

  def denormalize_from_0_to_1(self, actions):
    for k, v in actions.items():
      self.denormalized_actions[k] = scale_value(
          v, output_min=0, output_max=self.actions[k].max_value)
    return self.denormalized_actions
