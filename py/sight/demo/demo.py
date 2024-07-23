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
import inspect
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


def get_sight_instance():
  params = sight_pb2.Params(
      label='original_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    if(FLAGS.parent_id):
        sight_obj = sight_pb2.Object()
        # sight_obj.log_uid = str(sight.id)
        # sight_obj.set_attribute('log_uid', sight.id)
        sight_obj.sub_type = sight_pb2.Object.SubType.ST_LINK
        sight_obj.link.linked_sight_id = FLAGS.parent_id
        sight_obj.link.link_type = sight_pb2.Link.LinkType.LT_CHILD_TO_PARENT
        frame = inspect.currentframe().f_back.f_back.f_back
        sight.set_object_code_loc(sight_obj, frame)
        sight.log_object(sight_obj, True)
    with Block("A-block", sight):
      sight.text("A preText")
      with Attribute("key-Attribute", "A", sight):
        with Block("A1-B", sight):
          with Block("A1.1-B", sight):
            sight.text("A1.1 text")
      sight.text("A postText")

    # with Block("Block-B", sight):
    #   sight.text("B preText")
    #   with Attribute("key", "B", sight):
    #     with Attribute("key1", "B", sight):
    #       with Attribute("key2", "B", sight):
    #         with Attribute("key3", "B", sight):
    #           sight.text("B1 preText")
    #           with Block("block-B1", sight):
    #             with Block("block-B1.1", sight):
    #               sight.text("B1.1 text")
    #           sight.text("B1 postText")

    #         with Block("Block-B2", sight):
    #           with Attribute("keyin", "X", sight):
    #             with Attribute("keyin1", "X", sight):
    #               with Attribute("keyin2", "X", sight):
    #                 with Attribute("keyin3", "X", sight):
    #                   with Block("B2.1", sight):
    #                     sight.text("B2.1 text")

    #         with Block("Block-B3", sight):
    #           with Block("Block-B3.1", sight):
    #             sight.text("B3.1 text")
    #   sight.text("B postText")

    # with Block("C", sight):
    #   data = list(range(0, 60))
    #   data_structures.log(
    #       {
    #           "x": 1,
    #           (1, 2, 3): ["a", "b", {1: 2}],
    #           1: [1, 2],
    #           "1d": np.array(data),
    #           "2d": np.array(data).reshape((12, 5)),
    #           "3d": np.array(data).reshape((3, 4, 5)),
    #           "4d": np.array(data).reshape((3, 2, 2, 5)),
    #       },
    #       sight,
    #   )


if __name__ == "__main__":
  app.run(main)
