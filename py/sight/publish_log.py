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

"""Binary that takes a local Sight log file and publishes it remotely."""

from absl import app
from absl import flags
from google3.analysis.dremel.core.capacitor.public.python import pywrap_record_reader  # pylint:disable=line-too-long
from google3.googlex.cortex.sight.proto import sight_pb2
from google3.googlex.cortex.sight.py.sight import Sight
from google3.pyglib.contrib.gpathlib import gpath_flag

_LOG_FILE = gpath_flag.DEFINE_path(
    'log_file', None, 'File that contains the Sight log.', required=True)

_LOG_OWNER = flags.DEFINE_string(
    'log_owner', None, ('The owner of the published log.'), required=True)

_LOG_LABEL = flags.DEFINE_string('log_label', 'Sight Log',
                                 ('The descriptive label of this log.'))

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with Sight(
      sight_pb2.Params(
          label=_LOG_LABEL.value,
          log_owner=_LOG_OWNER.value,
          capacitor_output=True)) as sight:
    record_reader = pywrap_record_reader.RecordReader.CreateFromPath(
        _LOG_FILE.value, '*', 60)
    for obj in record_reader.IterRecords():
      sight.log_object(obj)


if __name__ == '__main__':
  app.run(main)
