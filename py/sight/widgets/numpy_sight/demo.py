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
"""Demo for the numpy bindings to the Sight logging library."""

from absl import app
from absl import flags
import numpy as np
from proto import sight_pb2
from py.sight import Sight
from py.widgets.numpy_sight import numpy_sight

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  params = sight_pb2.Params(
      label="Demo",
      # local=True,
      # text_output=True,
      log_owner="bronevet@google.com",
      capacitor_output=True,
      log_dir_path="/tmp/",
  )

  with Sight(params) as sight:
    a = np.array([[1, 2.2, 3.333], [4.1, 5, 6.2]], dtype=np.float32)
    for _ in range(0, 5):
      numpy_sight.log("a", a, sight)
      a = a * 2


if __name__ == "__main__":
  app.run(main)
