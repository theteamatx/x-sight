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

"""Individual simulation time steps in the Sight log."""

import inspect
from typing import Any, Sequence
from absl import logging

from proto import sight_pb2
from py.exception import exception


class SimulationTimeStep(object):
  """Encapsulates a single simulation time step un the Sight log."""

  def __init__(self, time_step_index: Sequence[int], time_step: float,
               time_step_size: float,
               time_step_units: sight_pb2.SimulationTimeStepStart.TimeStepUnits,
               sight: Any):
    """Creates and enters a simulation time step block.

    Args:
      time_step_index: Integral index of the time step within the overall
        ordering of time steps. The index can be hierarchical, supporting steps,
        sub-steps, etc., as appropriate. Indexes are ordered lexicographically.
      time_step: Exact value of time within the simulation.
      time_step_size: The amount of simulation time represented by this
        time-step.
      time_step_units: Units in which time is measured.
      sight: The logger instance to which the block is logged.

    Returns:
      The starting location of this time step block.
    """
    self.sight = sight
    if sight is None:
      logging.info('<<<SimulationTimeStep[ts_index=%s, ts=%s]', time_step_index,
                   time_step)
      return None

    if not self.sight.is_logging_enabled():
      return None

    # Register this simulation time step object with Sight.
    sight.widget_simulation_state.simulation_time_step = self

    if sight.widget_simulation_state.reference_trace:
      sight.widget_simulation_state.reference_trace.advance_to_within_block([
          sight_pb2.Object.ST_BLOCK_START,
          sight_pb2.BlockStart.ST_SIMULATION_TIME_STEP
      ])

    self.sight.set_attribute('SimulationTimeStep', str(time_step_index))
    # pytype: disable=attribute-error
    self.sight.enter_block(
        'SimulationTimeStep',
        sight_pb2.Object(
            block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_SIMULATION_TIME_STEP,
                simulation_time_step_start=sight_pb2.SimulationTimeStepStart(
                    time_step_index=time_step_index,
                    time_step=time_step,
                    time_step_size=time_step_size,
                    time_step_units=time_step_units,
                ))),
        inspect.currentframe().f_back.f_back)
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
      exception(exc_type, value, traceback, self.sight,
                inspect.currentframe().f_back)
      # pytype: enable=attribute-error

    if self.sight is None:
      logging.info('SimulationTimeStep>>>')
      return

    # Unregister this simulation time step object with Sight.
    if self.sight.widget_simulation_state.reference_trace:
      self.sight.widget_simulation_state.reference_trace.collect_current_block()
    self.sight.widget_simulation_state.simulation_time_step = None

    # pytype: disable=attribute-error
    self.sight.exit_block(
        'SimulationTimeStep',
        sight_pb2.Object(
            block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_SIMULATION_TIME_STEP)),
        inspect.currentframe().f_back)
    # pytype: enable=attribute-error
    self.sight.unset_attribute('SimulationTimeStep')
