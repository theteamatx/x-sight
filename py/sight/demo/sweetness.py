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

"""Demo of using the Sight Decision API to train an inverted pendulum controller."""

# import sys
# print("PYTHONPATH here is ", sys.path)

import random
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from sight.proto import sight_pb2
from sight import data_structures
from sight.sight import Sight
from sight.block import Block
from sight.widgets.decision import decision


def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
  print("In driver function of sweetness.........................")

  sweet_tooth = random.randrange(0, 10)
  data_structures.log_var('sweet_tooth', sweet_tooth, sight)

  for _ in range(1):
    choice = decision.decision_point('candy', sight)
    print("choice after dicision point: ", choice)

    logging.info('sweet_tooth=%s, choice=%s, joy=%s', sweet_tooth,
                 choice['sweetness'], float(choice['sweetness']) * sweet_tooth)
    
    decision.decision_outcome('joy', float(choice['sweetness']) * sweet_tooth, sight)
  print("Exiting driver function of sweetness........................")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  
  params = sight_pb2.Params(
      label="dummy_worker",
      log_owner="user@domain.com",
      local=True,
      text_output=True,
      capacitor_output=False,
      avro_output=True,
      log_dir_path="/tmp/",
      project_id = "cameltrain",
      bucket_name = "sight-meet",
      gcp_path = "workerData/",
      file_format = ".avro",
      dataset_name = "dsub",
      external_file_format = "AVRO",
      external_file_uri = "gs://"
      )

  with Sight(params) as sight:
    # with Block("worker", sight):
    decision.run(
        driver_fn=driver,
        state_attrs={
            'sweet_tooth': (0, 10),
        },
        action_attrs={
            'sweetness': (0, 10),
        },
        sight=sight)

if __name__ == '__main__':
  app.run(main)
