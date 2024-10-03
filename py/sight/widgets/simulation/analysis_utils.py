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
"""Utilities for analyzing Sight logs that document simulation runs."""

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple
)

from helpers.logs.logs_handler import logger as logging
import numpy as np
from sight import data_structures
from sight.proto import sight_pb2

Log = Sequence[sight_pb2.Object]
AnyObjMap = Dict[str, Any]
ObjMap = Dict[str, sight_pb2.Object]
KeyedObjMap = Tuple[str, Dict[str, Any]]


def single_objects_filter(obj: sight_pb2.Object,
                          sub_type: sight_pb2.Object.SubType) -> bool:
  return obj.sub_type == sub_type


def start_objects_filter(obj: sight_pb2.Object,
                         block_sub_type: sight_pb2.BlockStart.SubType) -> bool:
  return (obj.sub_type == sight_pb2.Object.ST_BLOCK_START and
          obj.block_start.sub_type == block_sub_type)


def log_uid(obj: sight_pb2.Object) -> str:
  for a in obj.attribute:
    if a.key != 'log_uid':
      continue
    return a.value
  return ''


def single_objects_key_parent(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    sub_type: sight_pb2.Object.SubType,
    label: str,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (object_col | 'single_objects_key_parent Filter ' + label >>
          beam.Filter(lambda obj: single_objects_filter(obj, sub_type)) |
          'single_objects_key_parent Map ' + label >> beam.Map(lambda x: (
              f'{x.ancestor_start_location[-2]} - {log_uid(x)}',
              {
                  label: x
              },
          )))


def single_objects_key_log_uid(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    sub_type: sight_pb2.Object.SubType,
    label: str,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (object_col | 'single_objects_key_log_uid Filter ' + label >>
          beam.Filter(lambda obj: single_objects_filter(obj, sub_type)) |
          'single_objects_key_log_uid Map ' + label >> beam.Map(lambda x:
                                                                (log_uid(x), {
                                                                    label: x
                                                                })))


def block_start_objects(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    block_sub_type: sight_pb2.BlockStart.SubType,
    label: str,
) -> beam.pvalue.PCollection[ObjMap]:
  return (object_col | 'objects Filter ' + label >>
          beam.Filter(lambda obj: start_objects_filter(obj, block_sub_type)) |
          'objects Map ' + label >> beam.Map(lambda x: ({
              label: x
          })))


def block_start_objects_key_self(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    block_sub_type: sight_pb2.BlockStart.SubType,
    label: str,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (object_col | 'objects_key_self Filter ' + label >>
          beam.Filter(lambda obj: start_objects_filter(obj, block_sub_type)) |
          'objects_key_self Map ' + label >>
          beam.Map(lambda x: (f'{x.location} - {log_uid(x)}', {
              label: x
          })))


def block_start_objects_key_parent(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    block_sub_type: sight_pb2.BlockStart.SubType,
    label: str,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (object_col | 'objects_key_parent Filter ' + label >>
          beam.Filter(lambda obj: start_objects_filter(obj, block_sub_type)) |
          'objects_key_parent Map ' + label >> beam.Map(lambda x: (
              f'{x.ancestor_start_location[-2]} - {log_uid(x)}',
              {
                  label: x
              },
          )))


def block_start_objects_key_log_uid(
    object_col: beam.pvalue.PCollection[sight_pb2.Object],
    block_sub_type: sight_pb2.BlockStart.SubType,
    label: str,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (
      object_col | 'objects_key_log_uid Filter ' + label >>
      beam.Filter(lambda obj: start_objects_filter(obj, block_sub_type)) |
      'objects_key_log_uid Map ' + label >> beam.Map(lambda x: (log_uid(x), {
          label: x
      })))


def create_constant_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return pcol | pcol_label + ' create_constant_key' >> beam.Map(lambda x:
                                                                ('', x))


def create_log_uid_key(
    pcol_label: str, new_key_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (pcol | pcol_label + ' ' + new_key_label + ' create_log_uid_key' >>
          beam.Map(lambda x: (log_uid(x[new_key_label]), x)))


def create_loc_log_uid_key(
    pcol_label: str, new_key_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (pcol | pcol_label + ' ' + new_key_label + ' create_loc_log_uid_key' >>
          beam.Map(lambda x: (
              f'{x[new_key_label].location} - {log_uid(x[new_key_label])}',
              x,
          )))


def create_named_value_label_log_uid_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return (pcol | pcol_label + ' create_named_value_label_log_uid_key' >>
          beam.Map(lambda x: (
              (f'{x["named_value"].block_start.label} -'
               f' {log_uid(x["named_value"])}'),
              x,
          )))


def create_var_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return pcol | pcol_label + ' create_var_key' >> beam.Map(lambda x:
                                                           (x['variable'], x))


def create_sim_ts_index_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[ObjMap]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return pcol | pcol_label + ' create_sim_ts_index_key' >> beam.Map(lambda x: (
      (f'{log_uid(x["simulation"])}-{x["simulation"].location} {x["simulation_time_step"].block_start.simulation_time_step_start.time_step_index[0]}'
      ),
      x,
  ))


def adjust_sim_ts_to_next_index_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[Dict[str, sight_pb2.Object]]
) -> beam.pvalue.PCollection[KeyedObjMap]:
  return pcol | pcol_label + ' adjust_sim_ts_to_next_index_key' >> beam.Map(
      lambda x: (f'{x[0].split()[0]} {int(x[0].split()[1]) + 1}', x[1]))


def remove_key(
    pcol_label: str, pcol: beam.pvalue.PCollection[KeyedObjMap]
) -> beam.pvalue.PCollection[ObjMap]:
  return pcol | pcol_label + ' remove_key' >> beam.Map(lambda x: x[1])


def change_key_to_self(
    pcol_label: str, obj_label: str, pcol: beam.pvalue.PCollection[KeyedObjMap]
) -> beam.pvalue.PCollection[ObjMap]:
  return pcol | pcol_label + ' change_key_to_self' >> beam.Map(lambda x: (
      f'{x[1][obj_label].location} - {log_uid(x[1][obj_label])}',
      x[1],
  ))


def change_key_to_parent(
    pcol_label: str, obj_label: str, pcol: beam.pvalue.PCollection[KeyedObjMap]
) -> beam.pvalue.PCollection[ObjMap]:
  return pcol | pcol_label + ' change_key_to_parent' >> beam.Map(lambda x: (
      (f'{x[1][obj_label].ancestor_start_location[-2]} -'
       f' {log_uid(x[1][obj_label])}'),
      x[1],
  ))


class ExtractAncestorBlockStartLocations(beam.DoFn):
  """Beam stage that extracts each object's ancestor context locations."""

  def process(self, obj: sight_pb2.Object) -> Iterator[KeyedObjMap]:
    """Extracts each object's ancestor context locations.

    Includes the starting point of the block's containing blocks and of the
    object ends a block, the starting point of that block.

    Args:
      obj: A Sight log object.

    Yields:
      Pairs with the starting point of each object's ancestral context block
      and the object itself.
    """
    for ancestor_start_location in obj.ancestor_start_location:
      yield (f'{ancestor_start_location} - {log_uid(obj)}', {'object': obj})

    if obj.sub_type == sight_pb2.Object.ST_BLOCK_END:
      yield (
          f'{obj.block_end.location_of_block_start} - {log_uid(obj)}',
          {
              'object': obj
          },
      )


class AddAncestorKeysToObjs(beam.DoFn):
  """Beam stage that extracts each object's ancestor context locations."""

  def __init__(self, anchor_obj_label: str):
    self.anchor_obj_label = anchor_obj_label

  def process(self, task: ObjMap) -> Iterator[KeyedObjMap]:
    """Attaches the ancestor locations of each object under .anchor_obj_label.

    Includes the starting point of the block's containing blocks and of the
    object ends a block, the starting point of that block.

    Args:
      task: Object map with one or more Sight Objects.

    Yields:
      Pairs with the starting point of each object's ancestral context block
      and the map itself.
    """
    obj = task[self.anchor_obj_label]
    for ancestor_start_location in obj.ancestor_start_location:
      yield (f'{ancestor_start_location} - {log_uid(obj)}', task)

    if obj.sub_type == sight_pb2.Object.ST_BLOCK_END:
      yield (f'{obj.block_end.location_of_block_start} - {log_uid(obj)}', task)


def objs_with_ancestor_keys(
    objects_map: KeyedObjMap,
    anchor_obj_label: str) -> beam.pvalue.PCollection[KeyedObjMap]:
  return remove_key(
      'objs_with_ancestor_keys ' + anchor_obj_label, objects_map
  ) | 'objs_with_ancestor_keys ' + anchor_obj_label >> beam.ParDo(
      AddAncestorKeysToObjs(anchor_obj_label))


class CombineRecords(beam.DoFn):
  """Combines CoGroupByKey-joined dicts from two sources."""

  def __init__(
      self,
      source1_label: str,
      source2_label: str,
  ):
    self.source1_label = source1_label
    self.source2_label = source2_label

  def process(
      self, task: Tuple[Any, Dict[str, Sequence[Dict[str, sight_pb2.Object]]]]
  ) -> Iterator[ObjMap]:
    """Combines CoGroupByKey-joined dicts from two sources.

    Args:
      task: Length <=1 sequences of dicts from two sources, indexed at labels
        source1_label and source2_label. The keys of the dicts from these
        sources are assumed to be disjoint.

    Yields:
      Unified dict that combines the key-value pairs from both sources.
      If the length of a given source is 0, its key-value pairs are not
      included in the output dict.
    """
    x: Dict[str, Sequence[ObjMap]] = task[1]
    source1: Sequence[ObjMap] = x[self.source1_label]
    if len(source1) > 1:
      logging.error(
          'Source 1 (%s) has %d entries, which is >1.',
          self.source1_label,
          len(source1),
      )
      return
    source2: List[ObjMap] = list(task[1][self.source2_label])
    if len(source2) > 1:
      logging.error(
          'Source 2 (%s) has %d entries, which is >1.',
          self.source2_label,
          len(source2),
      )
      return

    result: ObjMap = {}
    if source1:
      for key, val in source1[0].items():
        result[key] = val
    if source2:
      for key, val in source2[0].items():
        result[key] = val
    yield result


class ParentChildPairs(beam.DoFn):
  """Given a parent and a list of children, emits parent-child pairs.

  The key of these pairs is the location of the child object.

  Attributes:
    ancestors: Key of the ancestors object within the task dicts.
    child: Key of the child object within the task dicts.
    index_by_parent: Indicates whether the resulting object should be indexed by
      the location of the parent or the child object.
  """

  def __init__(
      self,
      ancestors: str,
      child: str,
      index_by_parent: bool,
  ):
    self.ancestors = ancestors
    self.child = child
    self.index_by_parent = index_by_parent

  def process(
      self, task: Tuple[str, Dict[str, Sequence[Dict[str, sight_pb2.Object]]]]
  ) -> Iterator[KeyedObjMap]:
    """Combines objects and their ancestors.

    Args:
      task: A pair of a key and - a sequence of ancestor log objects (assumed to
        always be size 1), and - a sequence of child child object that it
        contains.

    Yields:
      Tuples where the first element is the location of the child object and the
      second is a dictionary that contains all the ancestors and the child
      object.
    """
    ancestors_objs = task[1][self.ancestors]
    child_objs = task[1][self.child]

    # Skip named values that are not directly contained by ancestors.
    if not ancestors_objs:
      return

    if len(ancestors_objs) != 1:
      logging.error(
          ('Child objects cannot be contained within multiple ancestors!.'
           ' task=%s'),
          task,
      )
      return

    for child_obj in child_objs:
      cur = ancestors_objs[0].copy()
      for key in child_obj:
        if key not in cur:
          cur[key] = child_obj[key]
      if self.index_by_parent:
        location_idx = task[0]
      else:
        location_idx = (f'{child_obj[self.child].location} -'
                        f' {log_uid(child_obj[self.child])}')
      yield (location_idx, cur)


class SimulationStateNamedValuesToObjects(beam.DoFn):
  """Converts named value sub-logs within simulation containers into objects.

  Attributes:
    ancestors: Key of the ancestors object within the task dicts.
    value_objects: Key of the value_objects within the task dicts.
  """

  def __init__(
      self,
      ancestors: str,
      value_objects: str,
  ):
    self.ancestors = ancestors
    self.value_objects = value_objects

  def process(
      self, task: Tuple[str, Dict[str, Sequence[Dict[str, sight_pb2.Object]]]]
  ) -> Iterator[KeyedObjMap]:
    """Converts named value sub-logs within simulation containers into values.

    Args:
      task: A simulation container and the start of a named object, paired with
        all the log objects that describe the named object.

    Yields:
      Tuples where the first element is the location of the container object
      and the second maps the container and the value object.
    """
    # Skip named values that are not directly contained by a simulation
    # block (parameters or state).
    if not task[1][self.ancestors]:
      return

    if len(task[1][self.ancestors]) != 1:
      logging.error(
          ('Named values sub-logs cannot be contained within multiple named'
           ' values or containers!. task=%s'),
          task,
      )
      return

    if isinstance(task[1][self.ancestors][0], dict):
      log_and_obj: ObjMap = task[1][self.ancestors][0].copy()
    else:
      log_and_obj: ObjMap = {}
    log_and_obj['object'] = data_structures.from_log(
        [o['object'] for o in task[1][self.value_objects]])
    yield (
        (f'{log_and_obj["named_value"].location} -'
         f' {log_uid(log_and_obj["named_value"])}'),
        log_and_obj,
    )


class NamedObjectsToSequence(beam.DoFn):
  """Converts sets of named value objects to time-ordered sequences."""

  def process(self, task: Tuple[Any,
                                Iterable[AnyObjMap]]) -> Iterator[AnyObjMap]:
    """Time-orders the sequence of objects for a given simulation attribute.

    Args:
      task: A sequence of objects that describe the state of some simulation
        attribute over time.

    Yields:
      A time-ordered version of the input sequence.
    """
    ordered_seq = sorted(
        task[1],
        key=lambda x: list(x['simulation_time_step'].block_start.
                           simulation_time_step_start.time_step_index),
    )
    ts_indexes = np.array([
        x['simulation_time_step'].block_start.simulation_time_step_start.
        time_step_index for x in ordered_seq
    ],)
    time_steps = np.array([
        x['simulation_time_step'].block_start.simulation_time_step_start.
        time_step for x in ordered_seq
    ],)
    values = np.array([x['object'][1] for x in ordered_seq])

    yield {
        'simulation': ordered_seq[0]['simulation'],
        'cluster_id': ordered_seq[0].get('cluster_id'),
        'variable': ordered_seq[0]['named_value'].block_start.label,
        'values': values,
        'ts_indexes': ts_indexes,
        'time_steps': time_steps,
    }


class CombineParametersAndTimeSeries(beam.DoFn):
  """Combines the parameters and variable state time series of a simulation."""

  def __init__(
      self,
      params_label: str,
      variables_label: str,
  ):
    self.params_label = params_label
    self.variables_label = variables_label

  def process(self, task: Tuple[Any, Dict[str, List[ObjMap]]]) -> Iterator[Log]:
    """Combines the parameters and variable state time series of a simulation.

    Args:
      task: A sequence of objects that describe the state of some simulation
        attribute over time.

    Yields:
      A time-ordered version of the input sequence.
    """
    parameters = list(task[1][self.params_label])
    variables = list(task[1][self.variables_label])

    all_parameters = [p['object'] for p in parameters]

    for v in variables:
      res = v.copy()
      res['parameters'] = all_parameters
      yield res


def combine_parent_and_child(
    parent_label: str,
    parent_pcol: beam.pvalue.PCollection[KeyedObjMap],
    child_label: str,
    child_pcol: beam.pvalue.PCollection[KeyedObjMap],
    index_by_parent: bool,
) -> beam.pvalue.PCollection[KeyedObjMap]:
  """Joins a parent Objects to child Objects.

  Args:
    parent_label: identifies the parent PCollection.
    parent_pcol: contains KeyedObjMap that are keyed by the locations and
      log_uids of parent objects.
    child_label: identifies the child PCollection.
    child_pcol: contains KeyedObjMap that are keyed by the locations and
      log_uids of the immediate parents of child objects.
    index_by_parent: Indicates whether the resulting object should be indexed by
      the location of the parent or the child object.

  Returns:
    A PCollection that joins parent_pcol and child_pcol on their common key
    (the parent Object's location and log_uid). This collection is keyed by
    the location and log_uid of the child Object.
  """
  return ({
      parent_label: parent_pcol,
      child_label: child_pcol,
  } | parent_label + ' ' + child_label + ' CoGroupByKey' >> beam.CoGroupByKey()
          |
          parent_label + ' ' + child_label + ' ParentChildPairs' >> beam.ParDo(
              ParentChildPairs(parent_label, child_label, index_by_parent)))


def named_values_to_objects(
    parent_label: str,
    parent_pcol: beam.pvalue.PCollection[KeyedObjMap],
    child_label: str,
    objects_with_ancestors: beam.pvalue.PCollection[KeyedObjMap],
) -> beam.pvalue.PCollection[KeyedObjMap]:
  """Converts named value log regions into their corresponding Python objects.

  Args:
    parent_label: Unique label (among pipeline stages) for the collection of
      named_values.
    parent_pcol: ST_NAMED_VALUE objects.
    child_label: Unique label (among pipeline stages) for the collection of the
      children of ST_NAMED_VALUE objects.
    objects_with_ancestors: Objects, keyed by the start locations of any blocks
      that transitively contain them.

  Returns:
    Maps that contain the ST_NAMED_VALUES and their corresponding Python value
    objects and with the key of the ST_NAMED_VALUES Object.
  """
  return ({
      parent_label: parent_pcol,
      child_label: objects_with_ancestors,
  } | parent_label + ' ' + child_label + ' CoGroupByKey' >> beam.CoGroupByKey()
          | parent_label + ' ' + child_label +
          ' SimulationStateNamedValuesToObjects' >> beam.ParDo(
              SimulationStateNamedValuesToObjects(parent_label, child_label)))


def create_simulation_and_parameter_objects(
    log: beam.pvalue.PCollection[sight_pb2.Object],
    objects_with_ancestors: beam.pvalue.PCollection[KeyedObjMap],
    simulation: beam.pvalue.PCollection[KeyedObjMap],
    simulation_parameters: beam.pvalue.PCollection[KeyedObjMap],
    named_value: beam.pvalue.PCollection[KeyedObjMap],
    log_file_path_prefix: Optional[str],
) -> beam.pvalue.PCollection[AnyObjMap]:
  """Combines simulations and their parameter values.

  Args:
    log: All log objects.
    objects_with_ancestors: Objects, keyed by the start locations of any blocks
      that transitively contain them.
    simulation: All ST_SIMULATION objects.
    simulation_parameters: All ST_SIMULATION_PARAMETERS objects.
    named_value: All the ST_NAMED_VALUE objects.
    log_file_path_prefix: Prefix to use when writing intermediate states of the
      pipeline for debugging.

  Returns:
    AnyObjMaps that contain simulation objects, their contained simulation
      parameter objects, and the named values of those parameters.
  """
  simulations_and_parameters = combine_parent_and_child(
      'simulation',
      simulation,
      'simulation_parameters',
      simulation_parameters,
      index_by_parent=False,
  )

  simulation_parameters_and_named_values_key_named_value = (
      combine_parent_and_child(
          'simulations_and_parameters',
          simulations_and_parameters,
          'named_value',
          named_value,
          index_by_parent=False,
      ))

  return remove_key(
      'simulation_and_parameter_objects',
      named_values_to_objects(
          'simulation_parameters_and_named_values_key_named_value_objects',
          simulation_parameters_and_named_values_key_named_value,
          'objects',
          objects_with_ancestors,
      ),
  )


def create_simulation_states_params_and_named_value_objects(
    objects_with_ancestors: beam.pvalue.PCollection[KeyedObjMap],
    simulation_state: beam.pvalue.PCollection[KeyedObjMap],
    simulation_parameters: beam.pvalue.PCollection[KeyedObjMap],
    named_value: beam.pvalue.PCollection[KeyedObjMap],
    log_file_path_prefix: Optional[str],
) -> Tuple[beam.pvalue.PCollection[AnyObjMap],
           beam.pvalue.PCollection[AnyObjMap]]:
  """Combines simulation states and the named values within them.

  Args:
    objects_with_ancestors: Objects, keyed by the start locations of any blocks
      that transitively contain them.
    simulation_state: All ST_SIMULATION_STATE objects.
    simulation_parameters: All ST_SIMULATION_STATE objects.
    named_value: All the ST_NAMED_VALUE objects.
    log_file_path_prefix: Prefix to use when writing intermediate states of the
      pipeline for debugging.

  Returns:
    AnyObjMaps that contain simulation state objects and their associated named
    values.
  """
  named_value_objects = named_values_to_objects(
      'named_value',
      change_key_to_self('named_value_to_key_self', 'named_value', named_value),
      'objects',
      objects_with_ancestors,
  )
  named_value_objects_to_key_parent = change_key_to_parent(
      'named_value_objects_to_key_parent', 'named_value', named_value_objects)

  sim_state_named_values_key_state = combine_parent_and_child(
      'simulation_state',
      change_key_to_self('simulation_state_to_key_self', 'simulation_state',
                         simulation_state),
      'named_value',
      named_value_objects_to_key_parent,
      index_by_parent=True,
  )
  sim_params_named_values_key_params = combine_parent_and_child(
      'simulation_parameters',
      change_key_to_self(
          'simulation_parameters_to_key_self',
          'simulation_parameters',
          simulation_parameters,
      ),
      'named_value',
      named_value_objects_to_key_parent,
      index_by_parent=True,
  )
  return sim_state_named_values_key_state, sim_params_named_values_key_params


def create_simulation_time_step_state_objects(
    objects_with_ancestors: beam.pvalue.PCollection[KeyedObjMap],
    simulation: beam.pvalue.PCollection[KeyedObjMap],
    simulation_time_step: beam.pvalue.PCollection[KeyedObjMap],
    simulation_state: beam.pvalue.PCollection[KeyedObjMap],
    named_value: beam.pvalue.PCollection[KeyedObjMap],
    log_file_path_prefix: Optional[str],
) -> beam.pvalue.PCollection[AnyObjMap]:
  """Combines simulations and their time step values.

  Args:
    objects_with_ancestors: Objects, keyed by the start locations of any blocks
      that transitively contain them.
    simulation: All ST_SIMULATION objects.
    simulation_time_step: All ST_SIMULATION_TIME_STEP objects.
    simulation_state: All ST_SIMULATION_STATE objects.
    named_value: All the ST_NAMED_VALUE objects.
    log_file_path_prefix: Prefix to use when writing intermediate states of the
      pipeline for debugging.

  Returns:
    AnyObjMaps that contain simulation objects, their contained simulation
      time step objects, the simulation state objects within those and their
      associated named values.
  """
  named_value_objects = named_values_to_objects(
      'named_value',
      change_key_to_self('named_value_to_key_self', 'named_value', named_value),
      'objects',
      objects_with_ancestors,
  )

  # Connect simulation states to the named values logged within them.
  sim_state_named_values_key_state = combine_parent_and_child(
      'simulation_state',
      change_key_to_self('simulation_state_to_key_self', 'simulation_state',
                         simulation_state),
      'named_value',
      change_key_to_parent(
          'named_value_objects_to_key_parent',
          'named_value',
          named_value_objects,
      ),
      index_by_parent=True,
  )

  # Connect simulation time steps to their logged states and their named values.
  sim_ts_state_named_values_key_state = combine_parent_and_child(
      'simulation_time_step',
      change_key_to_self(
          'simulation_time_step_to_key_self',
          'simulation_time_step',
          simulation_time_step,
      ),
      'sim_state_named_values_key_state',
      change_key_to_parent(
          'sim_state_named_values_key_state_to_key_parent',
          'simulation_state',
          sim_state_named_values_key_state,
      ),
      index_by_parent=True,
  )

  # Connect simulations to their timesteps and logged states.
  sim_simul_ts_state_named_values_key_state = combine_parent_and_child(
      'simulation',
      change_key_to_self('simulation_to_key_self', 'simulation', simulation),
      'simulation_time_step',
      change_key_to_parent(
          'sim_ts_state_named_values_key_state_to_key_parent',
          'simulation_time_step',
          sim_ts_state_named_values_key_state,
      ),
      index_by_parent=True,
  )

  return remove_key(
      'sim_simul_ts_state_named_values_key_state',
      sim_simul_ts_state_named_values_key_state,
  )
