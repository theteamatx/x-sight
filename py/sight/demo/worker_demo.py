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
"""Demo for the python bindings to the Sight logging library."""

import inspect
import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sight import data_structures
from sight.widgets.decision import decision
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight

FLAGS = flags.FLAGS

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "worker_demo"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight:

    with Block("A-block", sight):
      sight.text("worker - A preText")
      with Attribute("key-Attribute", "A", sight):
        with Block("A1-B", sight):
          with Block("A1.1-B", sight):
            sight.text("worker - A1.1 text")
      sight.text("worker - A postText")

    with Block("Block-B", sight):
      sight.text("worker - B preText")
      with Attribute("key", "B", sight):

            with Block("Block-B3", sight):
              with Block("Block-B3.1", sight):
                sight.text("worker - B3.1 text")
      sight.text("worker - B postText")


if __name__ == "__main__":
  app.run(main)
