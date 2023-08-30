#!/usr/bin/python3
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


from typing import Any, Dict, List, Sequence, Tuple

import csv
from datetime import datetime
import math
import subprocess
from google.cloud import aiplatform

from absl import app
from absl import flags

_LOCAL = flags.DEFINE_boolean(
                              'local', False, 'Indicates whether the workers should run locally.')

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _LOCAL.value:
    provider='local' # For debugging... log files go in /tmp/dsub-local/{jobid}/task/...
  else:
    provider='google-cls-v2'

  PROJECT='cameltrain'
  LOGDIR='gs://dsub_cameltrain/sight/logging/'

  subprocess.run(['dsub', 
                  f'--provider={provider}',
                  '--regions=us-west1',
                  '--image=gcr.io/cameltrain/sight',
                  '--machine-type=e2-standard-2',
                  f'--project={PROJECT}',
                  f'--logging={LOGDIR}',
                  '--command=cd /project/x-sight/src && python3 py/demo/demo.py',
                  '--boot-disk-size=30',
                  '--tasks', 'tasks.tsv',
                  '--name', 'sight_test'])

if __name__ == '__main__':
  app.run(main)
