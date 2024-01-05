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

"""Analyze the impact of decisions on subsequent outcomes."""

import io
import math
import sys
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.coders import ProtoCoder
import joblib
import numpy as np
import pandas as pd
from proto import sight_pb2
from py.sight import Sight
from py.widgets.simulation import analysis_utils
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import capacitorio
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath_flag

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


class AnalyzeSequence(beam.DoFn):
  """Converts sets of named value objects to time-ordered sequences."""

  def __init__(
      self,
      named_value_and_object_label: str,
      decision_point_label: str,
      decision_outcome_label: str,
      configuration_label: str,
  ):
    self.named_value_and_object_label = named_value_and_object_label
    self.decision_point_label = decision_point_label
    self.decision_outcome_label = decision_outcome_label
    self.configuration_label = configuration_label

  def process(
      self,
      task: Tuple[
          Any, Dict[str, Union[List[Any], List[Dict[str, sight_pb2.Object]]]]
      ],
  ) -> Iterator[
      Tuple[
          str,
          Tuple[
              List[Tuple[Dict[str, Any], float]],
              Dict[str, sight_pb2.DecisionConfigurationStart.StateProps],
              Dict[str, sight_pb2.DecisionConfigurationStart.StateProps],
          ],
      ]
  ]:
    """Time-orders the sequence of objects for a given simulation attribute.

    Args:
      task: A sequence of objects that describe the state of some simulation
        attribute over time.

    Yields:
      A time-ordered version of the input sequence.
    """
    named_value_and_object = [
        (x['named_value'].location, x)
        for x in task[1][self.named_value_and_object_label]
    ]
    decision_point = [
        (x['decision_point'].location, x)
        for x in task[1][self.decision_point_label]
    ]
    decision_outcome = [
        (x['decision_outcome'].location, x)
        for x in task[1][self.decision_outcome_label]
    ]

    # Get the attributes used by the application within this simulation
    state_attrs = None
    action_attrs = None
    for cfg in task[1][self.configuration_label]:
      if (
          cfg['configuration'].block_start.configuration.sub_type
          == sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION
      ):
        if state_attrs:
          raise ValueError(
              'Multiple decision configurations present in run %s' % task[0]
          )
        decision_configuration = cfg[
            'configuration'
        ].block_start.configuration.decision_configuration
        state_attrs = decision_configuration.state_attrs
        action_attrs = decision_configuration.action_attrs

    if state_attrs is None:
      raise ValueError('No decision configuration present in run %s' % task[0])

    log = [
        x[1]
        for x in sorted(
            named_value_and_object + decision_point + decision_outcome,
            key=lambda x: x[0],
        )
    ]

    state = {}
    last_decision_point: sight_pb2.DecisionPoint = None
    accumulated_outcome = 0
    logging.info('state_attrs=%s', state_attrs)
    dataset: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}
    for obj in log:
      logging.info('obj=%s', obj)
      if 'object' in obj:
        if obj['object'][0] in state_attrs:
          state[obj['object'][0]] = obj['object'][1]
          logging.info('updated state=%s', state)
      elif 'decision_point' in obj:
        if last_decision_point:
          observation = last_decision_point_state.copy()
          for (
              param_name,
              param_value,
          ) in last_decision_point.choice_params.items():
            observation['chosen_param_' + param_name] = float(param_value)
          if last_decision_point.choice_label not in dataset:
            dataset[last_decision_point.choice_label] = []
          dataset[last_decision_point.choice_label].append(
              (observation, accumulated_outcome)
          )
          logging.info(
              'observation=%s, accumulated_outcome=%s, last_decision_point=%s',
              observation,
              accumulated_outcome,
              last_decision_point,
          )

        last_decision_point = obj['decision_point'].decision_point
        last_decision_point_state = state.copy()
        logging.info('last_decision_point_state=%s', last_decision_point_state)
        accumulated_outcome = 0
      elif 'decision_outcome' in obj:
        accumulated_outcome += float(
            obj['decision_outcome'].decision_outcome.outcome_value
        )
        logging.info(
            'outcome=%s', obj['decision_outcome'].decision_outcome.outcome_value
        )

    if last_decision_point:
      observation = last_decision_point_state.copy()
      for param_name, param_value in last_decision_point.choice_params.items():
        observation['chosen_param_' + param_name] = float(param_value)
      if last_decision_point.choice_label not in dataset:
        dataset[last_decision_point.choice_label] = []
      dataset[last_decision_point.choice_label].append(
          (observation, accumulated_outcome)
      )
      state = {}

    for choice_label, obs_data in dataset.items():
      yield (
          choice_label,
          (
              obs_data,
              state_attrs,
              action_attrs,
          ),
      )


class TrainOutcomePrediction(beam.DoFn):
  """Trains a model that predicts decision outcome values from decisions."""

  def process(
      self,
      task: Tuple[
          str,
          Iterable[
              Tuple[
                  List[Tuple[Dict[str, Any], float]],
                  Dict[str, sight_pb2.DecisionConfigurationStart.StateProps],
                  Dict[str, sight_pb2.DecisionConfigurationStart.StateProps],
              ]
          ],
      ],
  ) -> None:
    choice_label = task[0]
    columns = None

    state_attrs = None
    action_attrs = None
    for dataset in task[1]:
      if state_attrs is None:
        _, state_attrs, action_attrs = dataset
      else:
        if state_attrs != dataset[1] or action_attrs != dataset[2]:
          raise ValueError('Inconsistent state/action attributes across runs.')

    input_data = []
    output_data = []
    for dataset in task[1]:
      for obs in dataset[0]:
        if not columns:
          columns = obs[0].keys()
        row = []
        for c in columns:
          row.append(obs[0][c])
        input_data.append(row)
        output_data.append(obs[1])

    num_total_rows = len(input_data)
    num_train_rows = int(num_total_rows * 0.8)
    input_array = PolynomialFeatures(2).fit_transform(np.array(input_data))
    output_array = np.array(output_data)

    indices = np.random.permutation(num_total_rows)
    train_idx, eval_idx = indices[:num_train_rows], indices[num_train_rows:]
    train_input_data = input_array[train_idx, :]
    train_output_data = output_array[train_idx]
    eval_input_data = input_array[eval_idx, :]
    eval_output_data = output_array[eval_idx]

    np.set_printoptions(threshold=sys.maxsize)

    with gfile.Open(
        '/tmp/decision_outcomes.' + choice_label + '.csv', 'w'
    ) as f:
      pd.DataFrame(
          np.concatenate(
              (
                  input_array,
                  np.reshape(output_array, (output_array.shape[0], 1)),
              ),
              axis=1,
          )
      ).to_csv(f)

    lowest_error = 1e100
    best_model = None
    for learner in [
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        LinearRegression(),
    ]:
      model = learner.fit(train_input_data, train_output_data)

      predicted_array = model.predict(eval_input_data)

      logging.info(
          'eval_input_data%s=\n%s', eval_input_data.shape, eval_input_data
      )
      logging.info(
          'eval_output_data%s=\n%s', eval_output_data.shape, eval_output_data
      )
      logging.info(
          'predicted_array%s=%s', predicted_array.shape, predicted_array
      )
      mae = metrics.mean_absolute_error(eval_output_data, predicted_array)
      logging.info(
          '%s: mae=%s, rmse=%s',
          task[0],
          mae / abs(np.mean(eval_output_data)),
          math.sqrt(
              metrics.mean_squared_error(eval_output_data, predicted_array)
          )
          / abs(np.mean(eval_output_data)),
      )
      if lowest_error > mae:
        lowest_error = mae
        best_model = model

    with io.BytesIO() as model_bytes:
      joblib.dump(best_model, model_bytes)

      with Sight(
          sight_pb2.Params(
              label='Decision Outcomes',
              log_owner='bronevet@google.com',
              capacitor_output=True,
          )
      ) as sight:
        scikit_learn_algorithm = (
            sight_pb2.DecisionConfigurationStart.ScikitLearnAlgorithm()
        )
        scikit_learn_algorithm.model_encoding = model_bytes.getvalue()
        scikit_learn_algorithm.input_fields.extend(list(columns))

        choice_algorithm = (
            sight_pb2.DecisionConfigurationStart.ChoiceAlgorithm()
        )
        choice_algorithm.scikit_learn.CopyFrom(scikit_learn_algorithm)

        decision_configuration = sight_pb2.DecisionConfigurationStart()
        for attr_name, props in state_attrs.items():
          decision_configuration.state_attrs[attr_name].CopyFrom(props)
        for attr_name, props in action_attrs.items():
          decision_configuration.action_attrs[attr_name].CopyFrom(props)
        decision_configuration.choice_algorithm[choice_label].CopyFrom(
            choice_algorithm
        )

        sight.enter_block(
            'Decision Configuration',
            sight_pb2.Object(
                block_start=sight_pb2.BlockStart(
                    sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
                    configuration=sight_pb2.ConfigurationStart(
                        sub_type=sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION,
                        decision_configuration=decision_configuration,
                    ),
                )
            ),
        )
        sight.exit_block('Decision Configuration', sight_pb2.Object())


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root = beam.Pipeline(
      runner=runner.FlumeRunner()
  )  # beam.runners.DirectRunner())
  reads = []
  for file_path in _IN_LOG_FILE.value:
    reads.append(
        root
        | f'Read {file_path}'
        >> capacitorio.ReadFromCapacitor(
            file_path, ['*'], ProtoCoder(sight_pb2.Object)
        )
    )

  log: beam.pvalue.PCollection[sight_pb2.Object] = reads | beam.Flatten()

  objects_with_ancestors = log | beam.ParDo(
      analysis_utils.ExtractAncestorBlockStartLocations()
  )

  named_value = analysis_utils.block_start_objects_key_self(
      log, sight_pb2.BlockStart.ST_NAMED_VALUE, 'named_value'
  )
  decision_point = analysis_utils.single_objects_key_log_uid(
      log, sight_pb2.Object.ST_DECISION_POINT, 'decision_point'
  )
  decision_outcome = analysis_utils.single_objects_key_log_uid(
      log, sight_pb2.Object.ST_DECISION_OUTCOME, 'decision_outcome'
  )
  configuration = analysis_utils.block_start_objects_key_log_uid(
      log, sight_pb2.BlockStart.ST_CONFIGURATION, 'configuration'
  )

  _ = decision_point | 'decision_point' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.decision_point'
  )
  _ = decision_outcome | 'decision_outcome' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.decision_outcome'
  )

  named_value_and_object = analysis_utils.create_log_uid_key(
      'named_values_to_objects log_uid_key',
      'named_value',
      analysis_utils.named_values_to_objects(
          'named_value',
          named_value,
          'objects',
          objects_with_ancestors,
      ),
  )
  _ = named_value_and_object | 'named_value_and_object' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.named_value_and_object'
  )

  analyzed = (
      {
          'named_value_and_object': named_value_and_object,
          'decision_point': decision_point,
          'decision_outcome': decision_outcome,
          'configuration': configuration,
      }
      | 'named_value_and_object decision_point decision_outcome configuration CoGroupByKey'
      >> beam.CoGroupByKey()
      | 'named_value_and_object decision_point decision_outcome configuration AnalyzeSequence'
      >> beam.ParDo(
          AnalyzeSequence(
              'named_value_and_object',
              'decision_point',
              'decision_outcome',
              'configuration',
          )
      )
  )

  _ = analyzed | 'analyzed' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.analyzed'
  )

  _ = (
      analyzed
      | 'TrainOutcomePrediction GroupByKey' >> beam.GroupByKey()
      | 'TrainOutcomePrediction' >> beam.ParDo(TrainOutcomePrediction())
  )

  results = root.run()
  results.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
