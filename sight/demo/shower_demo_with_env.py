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

"""Demo of using the environment created using dm_env to train shower maintenance system."""

from typing import Sequence
from absl import app
from absl import flags
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision.acme import shower_env

FLAGS = flags.FLAGS

def get_sight_instance():
  params = sight_pb2.Params(
      label="shower_experiment",
      project_id=FLAGS.project_id,
      bucket_name=f"{FLAGS.project_id}-sight",
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(sight=sight, env=shower_env.ShowerEnv())


if __name__ == "__main__":
  app.run(main)
