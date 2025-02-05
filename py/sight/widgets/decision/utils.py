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
"""utility functions to be used in other functionalities."""

import yaml

def is_scalar(value):
  scalar_types = (int, float, str, bool, type(None), bytes)
  return isinstance(value, scalar_types)

def load_yaml_config(file_path):
  with open(file_path, 'r') as f:
    return yaml.safe_load(f)
