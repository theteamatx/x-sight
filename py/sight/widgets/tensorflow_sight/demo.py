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

"""Demo for the tensorflow bindings to the Sight logging library."""

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from proto import sight_pb2
from py.sight import Sight
from py.widgets.tensorflow_sight import tensorflow_sight

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  params = sight_pb2.Params(
      label="Demo",
      # local=True,
      # text_output=True,
      log_owner="user@domain.com",
      capacitor_output=True,
      log_dir_path="/tmp/")

  with Sight(params) as sight:
    with tensorflow_sight.TfModelTraining(label="Model Training", sight=sight):
      for epoch in range(0, 3):
        with tensorflow_sight.TfModelTrainingEpoch(
            label="Model Epoch", epoch_num=epoch, batch_size=10, sight=sight):
          sight.text("hello")
          with tensorflow_sight.TfModelApplication("Model Application", sight):
            a = np.array([[1 + epoch, 2.2 + epoch, 3.333 + epoch],
                          [4.1 + epoch, 5 + epoch, 6.2 + epoch]],
                         dtype=np.float32)
            for i in range(0, 5):
              tensorflow_sight.log(("tensor %d" % i), tf.convert_to_tensor(a),
                                   sight)
              a = a * 2


if __name__ == "__main__":
  app.run(main)
