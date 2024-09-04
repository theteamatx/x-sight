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

"""Trains a surrogate model to capture the observed dynamics of simulations."""

import math
import sys
from typing import Iterable, Iterator, List, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import capacitorio
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath_flag
from sight.proto import example_pb2
from sight.proto import sight_pb2

_IN_LOG_FILE = flags.DEFINE_list(
    'in_log_file',
    None,
    (
        'Input file(s) that contain the Sight log that documents the simulation'
        ' run.'
    ),
    required=True,
)

_OUT_FILE = gpath_flag.DEFINE_path(
    'out_file',
    None,
    'Input file that contains the Sight log that documents the simulation run.',
    required=True,
)

FLAGS = flags.FLAGS


class BigExamplesToSingleOutputRows(beam.DoFn):
  """Sight TensorFlowExamples Objects into one example with a single output."""

  def process(
      self, task: sight_pb2.Object
  ) -> Iterator[Tuple[str, Tuple[List[float], float]]]:
    """Chops up a Sight TensorflowExample Object into single-output examples.

    Each input object contains one or more input values and one or more output
    values. This method converts the inputs into a single row and yields a
    separate output object for each output variable.

    Args:
      task: A Sight TensorflowExample Object

    Yields:
      A time-ordered version of the input sequence.
    """
    # Turn the input example into a row.
    input_row = []
    if task.tensor_flow_example.input_example:
      for feat_name in task.tensor_flow_example.input_example.features.feature:
        # if feat_name in {'tai', 'fiald', 'dcph'}:
        input_row.append(
            task.tensor_flow_example.input_example.features.feature[
                feat_name
            ].float_list.value[0]
        )

    # Emit a single output for each output feature.
    if task.tensor_flow_example.output_example:
      for feat_key in task.tensor_flow_example.output_example.features.feature:
        yield (
            feat_key,
            (
                input_row,
                task.tensor_flow_example.output_example.features.feature[
                    feat_key
                ].float_list.value[0],
            ),
        )


class TrainModel(beam.DoFn):
  """Trains an ML model to predict outputs from inputs ."""

  def process(
      self, task: Tuple[str, Iterable[Tuple[List[float], float]]]
  ) -> Iterator[example_pb2.Example]:
    logging.info('TrainModel')
    input_data = []
    output_data = []
    for ex in task[1]:
      input_data.append(ex[0])
      output_data.append(ex[1])
    input_array = np.array(input_data)
    np.nan_to_num(input_array, copy=False)
    output_array = np.array(output_data)
    np.nan_to_num(output_array, copy=False)
    np.set_printoptions(threshold=sys.maxsize)
    learner = GradientBoostingRegressor()
    model = learner.fit(input_array, output_array)
    predicted_array = model.predict(input_array)
    logging.info(
        '%s: mae=%s, rmse=%s',
        task[0],
        metrics.mean_absolute_error(output_array, predicted_array)
        / np.mean(output_array),
        math.sqrt(metrics.mean_squared_error(output_array, predicted_array))
        / np.mean(output_array),
    )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root = beam.Pipeline(
      runner=runner.FlumeRunner()
  )  # beam.runners.DirectRunner())

  reads = []
  for file_path in _IN_LOG_FILE.value:
    if file_path.endswith('.txt'):
      with gfile.GFile(file_path, 'r') as inputs_f:
        for cur_file_path in inputs_f:
          logging.info('cur_file_path=%s', cur_file_path)
          reads.append(
              root
              | f'Read {cur_file_path}'
              >> capacitorio.ReadFromCapacitor(
                  cur_file_path, ['*'], beam.coders.ProtoCoder(sight_pb2.Object)
              )
          )
    else:
      logging.info('file_path=%s', file_path)
      reads.append(
          root
          | f'Read {file_path}'
          >> capacitorio.ReadFromCapacitor(
              file_path, ['*'], beam.coders.ProtoCoder(sight_pb2.Object)
          )
      )

    log = reads | beam.Flatten()
    _ = (
        log
        | beam.ParDo(BigExamplesToSingleOutputRows())
        | beam.GroupByKey()
        | beam.ParDo(TrainModel())
    )

  results = root.run()
  results.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
