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

"""Analyze Sight logs to track changes in simulation states across time steps."""

import ast
from typing import Any, Dict, Iterable, Iterator, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.coders import ProtoCoder
import pandas as pd

from proto import sight_pb2
from py.widgets.simulation import analysis_utils
from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import capacitorio
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath_flag
from proto import example_pb2
from proto import feature_pb2

_IN_LOG_FILE = flags.DEFINE_list(
    'in_log_file',
    None,
    'Input file(s) that contain the Sight log that documents the simulation run.',
    required=True)

_OUT_FILE = gpath_flag.DEFINE_path(
    'out_file',
    None,
    'Input file that contains the Sight log that documents the simulation run.',
    required=True)

FLAGS = flags.FLAGS


class SimulationTimeSeries(beam.DoFn):
  """Turns dicts from TimeStepPairDataRow into TensorFlow Example protos."""

  def process(self, task: Tuple[Any, Dict[str,
                                          Any]]) -> Iterator[Tuple[int, Any]]:
    if not task[1]['simulation']:
      return

    parameters = {}
    for param_named_val in task[1]['simulation_params_and_named_value_objects']:
      parameters[param_named_val['object'][0]] = param_named_val['object'][1]

    logging.info('parameters=%s', parameters)
    param_vars = sorted(parameters.keys())

    time_steps = {}
    for ts in task[1]['simulation_time_step']:
      time_steps[ts['simulation_time_step'].location] = {
          'simulation_time_step': ts['simulation_time_step'],
          'objects': {},
          'decision_point': None,
          'decision_outcome': None,
      }

    state_vars = set()
    for state_named_val in task[1]['simulation_states_and_named_value_objects']:
      for ancestor_loc in reversed(
          state_named_val['simulation_state'].ancestor_start_location):
        if ancestor_loc in time_steps:
          time_steps[ancestor_loc]['objects'][
              state_named_val['object'][0]] = state_named_val['object'][1]
          break
      state_vars.add(state_named_val['object'][0])
    state_vars = list(sorted(state_vars))

    decision_point_vars = None
    for dp in task[1]['decision_point']:
      for ancestor_loc in reversed(
          dp['decision_point'].ancestor_start_location):
        if ancestor_loc in time_steps:
          time_steps[ancestor_loc]['decision_point'] = dp['decision_point']
          break
      if decision_point_vars is None:
        decision_point_vars = dp[
            'decision_point'].decision_point.choice_params.keys()
    decision_point_vars = sorted(decision_point_vars)

    for do in task[1]['decision_outcome']:
      for ancestor_loc in reversed(
          do['decision_outcome'].ancestor_start_location):
        if ancestor_loc in time_steps:
          time_steps[ancestor_loc]['decision_outcome'] = do['decision_outcome']
          break

    last_decision_point_ts = None
    last_ts = None
    inputs_params = {v: [] for v in param_vars}
    inputs_state_vars = {v: [] for v in state_vars}
    inputs_decision = {v: [] for v in decision_point_vars}
    outputs_state_vars = {v: [] for v in state_vars}
    outputs_outcome = {v: [] for v in ['decision_outcome']}

    for loc in sorted(time_steps):
      ts = time_steps[loc]

      if last_decision_point_ts:
        logging.info(
            'choice_params=%s', last_decision_point_ts['decision_point']
            .decision_point.choice_params)

      if last_ts:
        cur_input = {}
        cur_output = {}
        for k in param_vars:
          inputs_params[k].append(parameters[k])
          cur_input[k] = parameters[k]
        for k in state_vars:
          inputs_state_vars[k].append(last_ts['objects'][k])
          cur_input[k] = last_ts['objects'][k]
          outputs_state_vars[k].append(ts['objects'][k])
          cur_output[k] = ts['objects'][k]
        for k in decision_point_vars:
          if last_decision_point_ts:
            inputs_decision[k].append(
                float(last_decision_point_ts['decision_point'].decision_point
                      .choice_params[k]))
            cur_input[k] = float(last_decision_point_ts['decision_point']
                                 .decision_point.choice_params[k])
          else:
            inputs_decision[k].append(0)
            cur_input[k] = 0
        outputs_outcome['decision_outcome'].append(
            ts['decision_outcome'].decision_outcome.outcome_value)
        cur_output[k] = ts['decision_outcome'].decision_outcome.outcome_value

        yield 0, (
            cur_input,
            cur_output,
        )

      if ts['decision_point']:
        last_decision_point_ts = ts
      last_ts = ts

    yield 1, {
        'inputs_params':
            pd.DataFrame(data=inputs_params, columns=param_vars),
        'inputs_state_vars':
            pd.DataFrame(data=inputs_state_vars, columns=state_vars),
        'inputs_decision':
            pd.DataFrame(data=inputs_decision, columns=decision_point_vars),
        'outputs_state_vars':
            pd.DataFrame(data=outputs_state_vars, columns=state_vars),
        'outputs_outcome':
            pd.DataFrame(data=outputs_outcome, columns=['decision_outcome']),
    }


class GatherAllTimeSeriesToCsv(beam.DoFn):
  """Turns dicts from TimeStepPairDataRow into TensorFlow Example protos."""

  def process(
      self, task: Tuple[int, Iterable[Dict[str, pd.DataFrame]]]
  ) -> Iterator[sight_pb2.Object]:
    logging.info('GatherAllTimeSeriesToCsv task=%s', task)
    for label in [
        'inputs_params', 'inputs_state_vars', 'inputs_decision',
        'outputs_state_vars', 'outputs_outcome'
    ]:
      data = pd.concat([df[label] for df in task[1]])
      with gfile.GFile(f'{str(_OUT_FILE.value)}.{label}.csv', 'w') as f:
        data.to_csv(f)


class TimeSeriesToTfExample(beam.DoFn):
  """Turns dicts from TimeStepPairDataRow into TensorFlow Example protos."""

  def _dict_to_ex(self, data: Dict[str, float]) -> example_pb2.Example:
    feature = feature_pb2.Features()
    for key, value in data.items():
      feature.feature[key].CopyFrom(
          feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[value])))
    return example_pb2.Example(features=feature)

  def process(
      self, task: Tuple[int, Iterable[Tuple[Dict[str, float], Dict[str,
                                                                   float]]]]
  ) -> Iterator[sight_pb2.Object]:
    for row in task[1]:
      obj = sight_pb2.Object(sub_type=sight_pb2.Object.ST_TENSORFLOW_EXAMPLE)
      tf_ex = sight_pb2.TensorFlowExample()
      tf_ex.input_example.CopyFrom(self._dict_to_ex(row[0]))
      tf_ex.output_example.CopyFrom(self._dict_to_ex(row[1]))
      obj.tensor_flow_example.CopyFrom(tf_ex)
      yield obj


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root = beam.Pipeline(
      runner=runner.FlumeRunner())  # beam.runners.DirectRunner())

  if gfile.Glob(str(_OUT_FILE.value) + '.simulation_time_series*'):
    simulation_time_series = (
        root
        | beam.io.textio.ReadFromText(
            str(_OUT_FILE.value) + '.simulation_time_series*')
        | beam.Map(ast.literal_eval))
  else:
    reads = []
    for file_path in _IN_LOG_FILE.value:
      if file_path.endswith('.txt'):
        with gfile.GFile(file_path, 'r') as inputs_f:
          for cur_file_path_line in inputs_f:
            cur_file_path = cur_file_path_line.decode()
            cur_file_path = cur_file_path.strip()
            logging.info('cur_file_path=%s (exists=%s)', cur_file_path,
                         gfile.Glob(cur_file_path))
            if gfile.Glob(cur_file_path):
              reads.append(
                  root
                  | f'Read {cur_file_path}' >> capacitorio.ReadFromCapacitor(
                      cur_file_path, ['*'], ProtoCoder(sight_pb2.Object)))
      else:
        if not file_path.endswith('.capacitor'):
          file_path = f'{file_path}.capacitor'
        logging.info('file_path=%s', file_path)
        reads.append(root
                     | f'Read {file_path}' >> capacitorio.ReadFromCapacitor(
                         file_path, ['*'], ProtoCoder(sight_pb2.Object)))

    log = reads | beam.Flatten()
    objects_with_ancestors = (
        log
        | beam.ParDo(analysis_utils.ExtractAncestorBlockStartLocations()))

    simulation = analysis_utils.block_start_objects_key_self(
        log, sight_pb2.BlockStart.ST_SIMULATION, 'simulation')
    simulation_parameters = analysis_utils.block_start_objects_key_parent(
        log, sight_pb2.BlockStart.ST_SIMULATION_PARAMETERS,
        'simulation_parameters')
    simulation_time_step = analysis_utils.objs_with_ancestor_keys(
        analysis_utils.block_start_objects_key_parent(
            log, sight_pb2.BlockStart.ST_SIMULATION_TIME_STEP,
            'simulation_time_step'), 'simulation_time_step')

    simulation_state = analysis_utils.block_start_objects_key_parent(
        log, sight_pb2.BlockStart.ST_SIMULATION_STATE, 'simulation_state')
    named_value = analysis_utils.block_start_objects_key_parent(
        log, sight_pb2.BlockStart.ST_NAMED_VALUE, 'named_value')

    decision_point = analysis_utils.objs_with_ancestor_keys(
        analysis_utils.single_objects_key_parent(
            log, sight_pb2.Object.ST_DECISION_POINT, 'decision_point'),
        'decision_point')
    _ = (
        decision_point
        | 'decision_point' >>
        beam.io.WriteToText(str(_OUT_FILE.value) + '.decision_point'))
    decision_outcome = analysis_utils.objs_with_ancestor_keys(
        analysis_utils.single_objects_key_parent(
            log, sight_pb2.Object.ST_DECISION_OUTCOME, 'decision_outcome'),
        'decision_outcome')
    _ = (
        decision_outcome
        | 'decision_outcome' >>
        beam.io.WriteToText(str(_OUT_FILE.value) + '.decision_outcome'))

    simulation_states_and_named_value_objects, simulation_params_and_named_value_objects = analysis_utils.create_simulation_states_params_and_named_value_objects(
        objects_with_ancestors,
        simulation_state,
        simulation_parameters,
        named_value,
        str(_OUT_FILE.value),
    )

    simulation_time_series = (
        {
            'simulation':
                simulation,
            'simulation_time_step':
                simulation_time_step,
            'decision_point':
                decision_point,
            'decision_outcome':
                decision_outcome,
            'simulation_states_and_named_value_objects':
                analysis_utils.objs_with_ancestor_keys(
                    simulation_states_and_named_value_objects,
                    'simulation_state'),
            'simulation_params_and_named_value_objects':
                analysis_utils.objs_with_ancestor_keys(
                    simulation_params_and_named_value_objects,
                    'simulation_parameters'),
        }
        | 'simulations_and_contents CoGroupByKey' >> beam.CoGroupByKey()
        | beam.ParDo(SimulationTimeSeries())
        | beam.GroupByKey())

    _ = (
        simulation_time_series
        | 'simulation_time_series' >>
        beam.io.WriteToText(str(_OUT_FILE.value) + '.simulation_time_series'))

  _ = (
      simulation_time_series
      | beam.Filter(lambda x: x[0] == 0)
      | beam.ParDo(TimeSeriesToTfExample())
      | capacitorio.WriteToCapacitor(
          str(_OUT_FILE.value) + '.examples', ProtoCoder(sight_pb2.Object)))

  _ = (
      simulation_time_series
      | beam.Filter(lambda x: x[0] == 1)
      | beam.ParDo(GatherAllTimeSeriesToCsv()))

  results = root.run()
  results.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
