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

"""Simulation state in the Sight log."""

import inspect
from typing import Any, Dict, Text
from absl import logging

from sight import data_structures
from sight.exception import exception
from sight.widgets.decision import decision
from sight.proto import sight_pb2


class SimulationState(object):
  """Encapsulates log region that documents a simulation's state."""

  def __init__(self, state: Dict[Text, Any], sight: Any) -> None:
    """Creates and enters a block of a simulation's state.

    Args:
      state: Key-value pairs that identify this block and all of its contained
        objects.
      sight: The logger instance to which the block is logged.

    Returns:
      The starting location of this simulation state block.
    """
    self.sight = sight
    if sight is None:
      logging.info('<<<SimulationState')
      return None

    if not self.sight.is_logging_enabled():
      return None

    # Register this simulation state object with Sight.
    sight.widget_simulation_state.simulation_state = self

    if sight.widget_simulation_state.reference_trace:
      sight.widget_simulation_state.reference_trace.advance_to_within_block([
          sight_pb2.Object.ST_BLOCK_START,
          sight_pb2.BlockStart.ST_SIMULATION_STATE,
      ])

    # pytype: disable=attribute-error
    self.sight.enter_block(
        'SimulationState',
        sight_pb2.Object(
            block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_SIMULATION_STATE
            )
        ),
        inspect.currentframe().f_back.f_back,
    )
    # pytype: enable=attribute-error

    for key, value in state.items():
      # pytype: disable=attribute-error
      data_structures.log_var(
          key, value, sight, inspect.currentframe().f_back.f_back
      )
      # pytype: enable=attribute-error

  def __enter__(self):
    return self

  def __exit__(self, exc_type, value, traceback):
    if not self.sight:
      return

    if not self.sight.is_logging_enabled():
      return

    if exc_type is not None:
      # pytype: disable=attribute-error
      exception(
          exc_type, value, traceback, self.sight, inspect.currentframe().f_back
      )
      # pytype: enable=attribute-error

    if self.sight is None:
      logging.info('SimulationState>>>')
      return

    # pytype: disable=attribute-error
    self.sight.exit_block(
        'SimulationState',
        sight_pb2.Object(
            block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_SIMULATION_STATE
            )
        ),
        inspect.currentframe().f_back,
    )
    # pytype: enable=attribute-error

    # If there is a reference trace, report the difference between
    # this trace and the reference trace via the Decision API.
    reference_trace = self.sight.widget_simulation_state.reference_trace
    if reference_trace:
      reference_state = {}
      while True:
        cur_named_var = reference_trace.advance_to_within_block([
            sight_pb2.Object.ST_BLOCK_START,
            sight_pb2.BlockStart.ST_NAMED_VALUE,
        ])
        if not cur_named_var:
          break
        name, value = data_structures.from_ordered_log(
            reference_trace.collect_current_block()
        )
        reference_state[name] = value

      observed_state_vars = reference_state.keys()
      sum_relative_errors = 0
      num_vars = 0
      for name in observed_state_vars:
        if (
            max(
                abs(self.sight.widget_simulation_state.state[name]),
                abs(reference_state[name]),
            )
            > 0
        ):
          sum_relative_errors += abs(
              (
                  self.sight.widget_simulation_state.state[name]
                  - reference_state[name]
              )
              / max(
                  abs(self.sight.widget_simulation_state.state[name]),
                  abs(reference_state[name]),
              )
          )
          num_vars += 1

      error_relative_to_reference_run = (
          sum_relative_errors / num_vars if num_vars > 0 else 0
      )
      decision.decision_outcome(
          'distance', 0 - error_relative_to_reference_run, self.sight
      )

    # Unregister this simulation state object with Sight.
    if self.sight.widget_simulation_state.reference_trace:
      self.sight.widget_simulation_state.reference_trace.collect_current_block()
    self.sight.widget_simulation_state.simulation_state = None


def state_updated(
    name: str,
    obj_to_log: Any,
    sight: Any,
) -> None:
  """Informs the Simulation API that the current state has been updated.

  Args:
    name: The name of the updated state variable.
    obj_to_log: The value of the state variable.
    sight: Instance of a Sight logger.
  """
  if (
      sight.widget_simulation_state
      and sight.widget_simulation_state.simulation_state
  ):
    sight.widget_simulation_state.state[name] = obj_to_log
