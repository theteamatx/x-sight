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

"""Launches XManager workers that run the Vizier optimization."""

from typing import Sequence

from absl import app
from absl import flags

from xmanager import xm
from xmanager import xm_abc

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'The name of the vizier experiments.',
    required=True,
)

_BINARY_PATH = flags.DEFINE_string(
    'binary_path',
    None,
    (
        'Path of the Blaze target of the binary that contains the code to be'
        ' configured.'
    ),
    required=True,
)

_REFERENCE_RUN_FILE = flags.DEFINE_string(
    'reference_run_file',
    None,
    'File that contains the Sight log of a reference run.',
)

_CELL = flags.DEFINE_string(
    'cell', 'is', 'The Borg cell where the experiment will run'
)

_NUM_TRAIN_WORKERS = flags.DEFINE_integer(
    'num_train_workers', 1, 'Number of workers to use in a training run.'
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with xm_abc.create_experiment(
      experiment_title=_EXPERIMENT_NAME.value
  ) as experiment:
    [executable] = experiment.package([
        xm.bazel_binary(
            label=_BINARY_PATH.value, executor_spec=xm_abc.Borg.Spec()
        ),
    ])

    job = xm.Job(
        executable=executable,
        args={
            'decision_mode': 'train',
            'decision_train_alg': 'vizier_local',
            'vizier_train_experiment_name': _EXPERIMENT_NAME.value,
            'reference_run_file': _REFERENCE_RUN_FILE.value,
        },
        executor=xm_abc.Borg(
            xm.JobRequirements(
                cpu=1 * xm.vCPU,
                ram=32 * xm.GiB,
                replicas=_NUM_TRAIN_WORKERS.value,
                service_tier=xm.ServiceTier.BATCH,
                location=_CELL.value,
            )
        ),
    )
    experiment.add(job)


if __name__ == '__main__':
  app.run(main)
