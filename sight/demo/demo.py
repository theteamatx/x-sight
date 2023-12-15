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

import os
from absl import app
from absl import flags
import numpy as np
from sight import data_structures
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight

FLAGS = flags.FLAGS


def get_sight_instance():
  params = sight_pb2.Params(
      label='demo file',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    with Block("A", sight):
      sight.text("A preText")
      with Attribute("key", "A", sight):
        with Block("A1", sight):
          with Block("A1.1", sight):
            sight.text("A1.1 text")
      sight.text("A postText")

    with Block("B", sight):
      sight.text("B preText")
      with Attribute("key", "B", sight):
        with Attribute("key1", "B", sight):
          with Attribute("key2", "B", sight):
            with Attribute("key3", "B", sight):
              sight.text("B1 preText")
              with Block("B1", sight):
                with Block("B1.1", sight):
                  sight.text("B1.1 text")
              sight.text("B1 postText")

            with Block("B2", sight):
              with Attribute("keyin", "X", sight):
                with Attribute("keyin1", "X", sight):
                  with Attribute("keyin2", "X", sight):
                    with Attribute("keyin3", "X", sight):
                      with Block("B2.1", sight):
                        sight.text("B2.1 text")

            with Block("B3", sight):
              with Block("B3.1", sight):
                sight.text("B3.1 text")
      sight.text("B postText")

    with Block("C", sight):
      data = list(range(0, 60))
      data_structures.log(
          {
              "x": 1,
              (1, 2, 3): ["a", "b", {1: 2}],
              1: [1, 2],
              "1d": np.array(data),
              "2d": np.array(data).reshape((12, 5)),
              "3d": np.array(data).reshape((3, 4, 5)),
              "4d": np.array(data).reshape((3, 2, 2, 5)),
          },
          sight,
      )


if __name__ == "__main__":
  app.run(main)
