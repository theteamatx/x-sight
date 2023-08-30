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
from absl import logging

from sight.proto import sight_pb2
from sight import data_structures
from sight.sight import Sight
from sight.block import Block
from sight.widgets.decision import decision

_file_name = 'fractal_demo.py'

def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
  method_name = "driver"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  print("In driver function of fractal_demo.........................")
  # raise SystemExit(0)
  agent_carrier_0_c0_revenue = random.randrange(0, 10000)
  data_structures.log_var('agent_carrier_0_c0_revenue', agent_carrier_0_c0_revenue, sight)

  agent_carrier_0_c0_profit = random.randrange(0, 10000)
  data_structures.log_var('agent_carrier_0_c0_profit', agent_carrier_0_c0_profit, sight)

  agent_shipper_0_c0_fee_to_shipper = random.randrange(0, 10000)
  data_structures.log_var('agent_shipper_0_c0_fee_to_shipper', agent_shipper_0_c0_fee_to_shipper, sight)
  
  agent_shipper_0_c0_total_shipment_count = random.randrange(0, 10000)
  data_structures.log_var('agent_shipper_0_c0_total_shipment_count', agent_shipper_0_c0_total_shipment_count, sight)

  for _ in range(3): #simulated timesteps
    choice = decision.decision_point('DP_label', sight)
    # raise SystemExit(0)
    logging.info('choice["unreliability"]=%s, joy=%s', 
                 choice['unreliability'], float(choice['unreliability']) * agent_carrier_0_c0_revenue)
    decision.decision_outcome('DO_label',
                              float(choice['unreliability']) * agent_carrier_0_c0_revenue, sight)
  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")


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
            'agent_carrier_0_c0_revenue': (0, 10000),
            'agent_carrier_0_c0_profit': (0, 10000),
            'agent_shipper_0_c0_fee_to_shipper': (0, 10000),
            'agent_shipper_0_c0_total_shipment_count': (0, 10000),
        },
        action_attrs={
            'unreliability': (0, 10000)
        },
        sight=sight)


if __name__ == '__main__':
  app.run(main)
