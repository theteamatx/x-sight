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
"""Demo of an application that streams its output over time."""

import math
import os
import time

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sight import data_structures
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with Sight(
      sight_pb2.Params(
          label='demo file',
          bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
      )) as sight:
    for ts in range(1000):
      print('ts=%s' % ts)
      with Block("Time step", '%08d' % ts, sight):
        sight.text('Time!')
        data_structures.log_var(
            'state',
            pd.DataFrame(
                {'Val': [math.sin(math.radians(ts + i)) for i in range(100)]}),
            sight)
      time.sleep(5)


if __name__ == "__main__":
  app.run(main)
