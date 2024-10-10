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
"""Extract and log the sequences for each value in Sight logs."""

from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from absl import app
from absl import flags
import apache_beam as beam
from google3.analysis.dremel.core.capacitor.public.python import (
    pywrap_record_reader
)
from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import capacitorio
from google3.pyglib import gfile
from google3.pyglib.contrib.gpathlib import gpath_flag
from helpers.logs.logs_handler import logger as logging
import numpy as np
from sight import data_structures
from sight.attribute import Attribute
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.numpy_sight import numpy_sight
from sight.widgets.simulation import analysis_utils

_IN_CLUSTERS_FILE = gpath_flag.DEFINE_path(
    'in_clusters_file',
    None,
    ('Input file that contains the Sight log that documents how the '
     'simulation runs are clustered.'),
    required=False,
)

_NUM_CLUSTERS = flags.DEFINE_integer(
    'num_clusters', 0, 'The number of clusters to divide the runs into')

_IN_LOG_FILE = flags.DEFINE_list(
    'in_log_file',
    None,
    ('Input file(s) that contain the Sight log that documents the simulation'
     ' run.'),
    required=True,
)

_OUT_FILE = gpath_flag.DEFINE_path(
    'out_file',
    None,
    'Input file that contains the Sight log that documents the simulation run.',
    required=True,
)

FLAGS = flags.FLAGS


class LogVarSequence(beam.DoFn):
  """Converts sets of named value objects to time-ordered sequences."""

  def __init__(
      self,
      file_name_prefix: str,
  ):
    self.file_name_prefix = file_name_prefix

  def process(
      self, task: Tuple[Any, Iterable[Any]]
  ) -> Iterator[Tuple[str, Tuple[str, List[numpy_sight.LabeledNpArray]]]]:
    """Time-orders the sequence of objects for a given simulation attribute.

    Args:
      task: A sequence of objects that describe the state of some simulation
        attribute over time.

    Yields:
      A time-ordered version of the input sequence.
    """
    variables = list(task[1])
    logging.info('LogVarSequence task=%s', task)
    logging.info('LogVarSequence variables=%s', list(task[1]))

    # Group variables by their cluster
    cluster_to_variables = {}
    for v in variables:
      cluster_id = data_structures.from_ordered_log([v['cluster_id']])
      if cluster_id not in cluster_to_variables:
        cluster_to_variables[cluster_id] = []
      cluster_to_variables[cluster_id].append(v)

    logging.info('LogVarSequence cluster_to_variables=%s', cluster_to_variables)
    cluster_vars = []
    for cluster_id in sorted(cluster_to_variables):
      cluster_variables = cluster_to_variables[cluster_id]
      all_parameter_values = {}
      for v in cluster_variables:
        for p in v['parameters']:
          if p[0] not in all_parameter_values:
            all_parameter_values[p[0]] = set()
          all_parameter_values[p[0]].add(p[1])
      logging.info('all_parameter_values=%s', all_parameter_values)

      varied_parameters = set()
      for param_name, values in all_parameter_values.items():
        if len(values) > 1:
          varied_parameters.add(param_name)
      logging.info('varied_parameters=%s', varied_parameters)
      ordered_varied_parameters = sorted(varied_parameters)

      all_time_steps = {}
      for v in cluster_variables:
        logging.info('v["time_steps"]=%s', v['time_steps'])
        for ts in v['time_steps']:
          if ts not in all_time_steps:
            all_time_steps[ts] = 0
          all_time_steps[ts] += 1
      logging.info('all_time_steps=%s', all_time_steps)

      for v in cluster_variables:
        logging.info(
            '%s: v["values"](%s)=%s',
            v['parameters'],
            v['values'].shape,
            v['values'],
        )

      all_values = np.array([v['values'] for v in cluster_variables])
      logging.info('all_values(%s)=%s', all_values.shape, all_values)

      var_params = []
      for v in cluster_variables:
        var_params.append(', '.join([
            f'{p}={dict(v["parameters"])[p]}' for p in ordered_varied_parameters
        ]))

      cluster_vars.append(
          numpy_sight.LabeledNpArray(
              all_values,
              ['params', 'time_steps'],
              [var_params, [str(i) for i in all_time_steps.keys()]],
          ))

    yield ('', (task[0], cluster_vars))


class AggregateLogs(beam.DoFn):
  """Collects the logs for multiple variables into a single log."""

  def process(
      self,
      task: Tuple[str, Iterable[Tuple[str, List[numpy_sight.LabeledNpArray]]]],
  ) -> None:
    """Time-orders the sequence of objects for a given simulation attribute.

    Args:
      task: A sequence of objects that describe the state of some simulation
        attribute over time.
    """
    with Sight(
        sight_pb2.Params(
            label='Simulation',
            log_owner='bronevet@google.com',
            capacitor_output=True,
            log_dir_path='/tmp/',
        )) as sight:
      for var in task[1]:
        logging.info('AggregateLogs: var=%s', var)
        with Attribute('variable', var[0], sight):
          var_clusters = var[1]
          for i in range(len(var_clusters)):
            with Attribute('cluster', str(i), sight):
              data_structures.log_var(var[0], var_clusters[i], sight)


def read_capacitor_file(filename: str,
                        fields: Sequence[str] = ('*',),
                        timeout: float = 60.0) -> Iterator[Any]:
  """Yields all records from a capacitor file.

  Args:
    filename: May be single file, or  pattern.
    fields: Subset of fields to read. Default is to read all fields.
    timeout: I/O timeout.
  """
  filenames = gfile.Glob(filename)
  if not filenames:
    raise ValueError(f'No such file: {filename}')
  for filename in filenames:
    reader = pywrap_record_reader.RecordReader.CreateFromPath(
        filename, fields, timeout)
    for r in reader.IterRecords():
      yield r


def sight_encode_value(val: int) -> sight_pb2.Object:
  """Encodes a value as a Sight object.

  This is done to ensure type consistency among the many data structures
  being used to describe simulation behavior.

  Args:
    val: The value to be encoded.

  Returns:
    The single Sight log object that encodes val.
  """
  with Sight(sight_pb2.Params(capacitor_output=True, in_memory=True)) as sight:
    data_structures.log(val, sight)
    return sight.get_in_memory_log().obj[0]


def load_log_uid_clusters(
    root: beam.Pipeline, simulation_log_uid: beam.pvalue.PCollection
) -> beam.pvalue.PCollection[Tuple[str, Dict[str, sight_pb2.Object]]]:
  """Loads clusters of logs into a PCollection.

  Args:
    root: The Beam pipeline.
    simulation_log_uid: The PCollection that contains all the log ids. Used if
      the _IN_CLUSTERS_FILE command line flag is not specified.

  Returns:
    PCollection of mappings from each log unique id to the unique id of the
    cluster that log was assigned to.
  """

  if _IN_CLUSTERS_FILE.value:
    cluster_assignment = {}
    for clusters_fname in gfile.Glob(_IN_CLUSTERS_FILE.value):
      for message in read_capacitor_file(
          clusters_fname,
          [
              '*',
          ],
          60,
      ):
        cluster_assignment_log = sight_pb2.Log()
        cluster_assignment_log.ParseFromString(message.SerializeToString())
        cluster_assignment = data_structures.from_log(
            list(cluster_assignment_log.obj))
        if cluster_assignment['num_clusters'] == _NUM_CLUSTERS.value:
          for key, value in data_structures.from_log(
              list(cluster_assignment_log.obj)).items():
            cluster_assignment[key] = value
          break
    if not cluster_assignment:
      logging.error('Failed to find a clustering with %d clusters.',
                    _NUM_CLUSTERS.value)
      return
    logging.info('cluster_assignment=%s', cluster_assignment)

    log_to_cluster_id = []
    for log_uid, cluster_id in cluster_assignment['cluster_assignment'].items():
      log_to_cluster_id.append((log_uid, {
          'cluster_id': sight_encode_value(cluster_id)
      }))
    return root | beam.Create(log_to_cluster_id)
  else:
    return simulation_log_uid | beam.Map(lambda x: (x[0], {
        'cluster_id': sight_encode_value(0)
    }))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root = beam.Pipeline(
      runner=runner.FlumeRunner())  # beam.runners.DirectRunner())
  reads = []
  for file_path in _IN_LOG_FILE.value:
    reads.append(root | f'Read {file_path}' >> capacitorio.ReadFromCapacitor(
        file_path, ['*'], beam.coders.ProtoCoder(sight_pb2.Object)))

  log: beam.pvalue.PCollection[sight_pb2.Object] = reads | beam.Flatten()

  simulation = analysis_utils.objects(log, sight_pb2.BlockStart.ST_SIMULATION,
                                      'simulation')

  simulation_log_uid: beam.pvalue.PCollection[analysis_utils.KeyedObjMap] = (
      analysis_utils.create_log_uid_key('simulation_and_cluster', 'simulation',
                                        simulation))

  clusters_key_log_uid = load_log_uid_clusters(root, simulation_log_uid)
  _ = clusters_key_log_uid | 'clusters_key_log_uid' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.clusters_key_log_uid')

  objects_with_ancestors = log | beam.ParDo(
      analysis_utils.ExtractAncestorBlockStartLocations())

  simulation_and_cluster: beam.pvalue.PCollection[analysis_utils.ObjMap] = ({
      'simulation': simulation_log_uid,
      'clusters_key_log_uid': clusters_key_log_uid,
  } | 'simulation_and_cluster CoGroupByKey' >> beam.CoGroupByKey() | beam.ParDo(
      analysis_utils.CombineRecords('simulation', 'clusters_key_log_uid')))
  simulation_and_cluster_sim_loc_uid: beam.pvalue.PCollection[
      analysis_utils.KeyedObjMap] = analysis_utils.create_loc_log_uid_key(
          'simulation_and_cluster', 'simulation', simulation_and_cluster)

  _ = (simulation_and_cluster | 'write simulation_and_cluster' >>
       beam.io.WriteToText(str(_OUT_FILE.value) + '.simulation_and_cluster'))
  _ = (simulation_and_cluster_sim_loc_uid |
       'write simulation_and_cluster_sim_loc_uid' >> beam.io.WriteToText(
           str(_OUT_FILE.value) + '.simulation_and_cluster_sim_loc_uid'))

  simulation_parameters = analysis_utils.block_start_objects_key_parent(
      log,
      sight_pb2.BlockStart.ST_SIMULATION_PARAMETERS,
      'simulation_parameters',
  )
  _ = (simulation_parameters | 'write simulation_parameters' >>
       beam.io.WriteToText(str(_OUT_FILE.value) + '.simulation_parameters'))
  simulation_time_step = analysis_utils.block_start_objects_key_parent(
      log, sight_pb2.BlockStart.ST_SIMULATION_TIME_STEP, 'simulation_time_step')
  simulation_state = analysis_utils.block_start_objects_key_parent(
      log, sight_pb2.BlockStart.ST_SIMULATION_STATE, 'simulation_state')
  named_value = analysis_utils.block_start_objects_key_parent(
      log, sight_pb2.BlockStart.ST_NAMED_VALUE, 'named_value')
  _ = simulation_time_step | 'simulation_time_step' >> beam.io.WriteToText(
      str(_OUT_FILE.value) + '.simulation_time_step')

  simulation_and_parameter_objects = (
      analysis_utils.create_simulation_and_parameter_objects(
          log,
          objects_with_ancestors,
          simulation_and_cluster_sim_loc_uid,
          simulation_parameters,
          named_value,
          str(_OUT_FILE.value),
      ))
  _ = (simulation_and_parameter_objects |
       'simulation_and_parameter_objects' >> beam.io.WriteToText(
           str(_OUT_FILE.value) + '.simulation_and_parameter_objects'))

  simulation_time_step_state_objects = (
      analysis_utils.create_simulation_time_step_state_objects(
          objects_with_ancestors,
          simulation_and_cluster_sim_loc_uid,
          simulation_time_step,
          simulation_state,
          named_value,
          str(_OUT_FILE.value),
      ))
  _ = (simulation_time_step_state_objects |
       'simulation_time_step_state_objects' >> beam.io.WriteToText(
           str(_OUT_FILE.value) + '.simulation_time_step_state_objects'))

  simulation_time_step_state_objects_in_time_order = (
      analysis_utils.create_named_value_label_log_uid_key(
          'simulation_time_step_state_objects',
          simulation_time_step_state_objects,
      ) | 'simulation_time_step_state_objects GroupByKey' >> beam.GroupByKey() |
      beam.ParDo(analysis_utils.NamedObjectsToSequence()))

  _ = (
      simulation_time_step_state_objects_in_time_order |
      'simulation_time_step_state_objects_in_time_order' >> beam.io.WriteToText(
          str(_OUT_FILE.value) +
          '.simulation_time_step_state_objects_in_time_order'))

  simulation_params_and_vars = (
      {
          'simulation_and_parameter_objects':
              (analysis_utils.create_loc_log_uid_key(
                  'simulation_and_parameter_objects',
                  'simulation',
                  simulation_and_parameter_objects,
              )),
          'simulation_time_step_state_objects_in_time_order':
              (analysis_utils.create_loc_log_uid_key(
                  'simulation_time_step_state_objects_in_time_order',
                  'simulation',
                  simulation_time_step_state_objects_in_time_order,
              )),
      } | 'simulation_params_steps_objects_in_time_order CoGroupByKey' >>
      beam.CoGroupByKey() | beam.ParDo(
          analysis_utils.CombineParametersAndTimeSeries(
              'simulation_and_parameter_objects',
              'simulation_time_step_state_objects_in_time_order',
          )))

  _ = (
      simulation_params_and_vars | 'simulation_params_and_vars' >>
      beam.io.WriteToText(str(_OUT_FILE.value) + '.simulation_params_and_vars'))

  _ = (analysis_utils.create_var_key('simulation_params_and_vars',
                                     simulation_params_and_vars) |
       'simulation_params_and_vars GroupByKey - varname' >> beam.GroupByKey() |
       beam.ParDo(LogVarSequence(str(_OUT_FILE.value))) |
       'log var sequences gather all' >> beam.GroupByKey() |
       beam.ParDo(AggregateLogs()))

  results = root.run()
  results.wait_until_finish()


if __name__ == '__main__':
  app.run(main)
