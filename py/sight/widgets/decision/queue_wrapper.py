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


"""wrapper class to be used to access global data structure."""

import queue

# singleton class
class QueueDictWrapper:

    # Shared with every instance
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        self.queue = queue.Queue()
        self.queue.put({})

    def get(self):
      central_dict = self.queue.get()
      self.queue.put(central_dict)
      self.queue.task_done()
      return central_dict

    def get_for_key(self, key):
      central_dict = self.queue.get()
      self.queue.put(central_dict)
      self.queue.task_done()
      return central_dict.get(key,None)

    def set_for_key(self, key, value):
      central_dict = self.queue.get()
      central_dict.update({key:value})
      self.queue.put(central_dict)
      self.queue.task_done()

    def update(self, mapping):
      central_dict = self.queue.get()
      central_dict.update(mapping)
      self.queue.put(central_dict)
      self.queue.task_done()
