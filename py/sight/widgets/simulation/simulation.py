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
"""Simulation runs in the Sight log."""

import inspect
from typing import Any, Callable, Dict, Optional, Text, Tuple

from helpers.logs.logs_handler import logger as logging
from sight.exception import exception
from sight.proto import sight_pb2
from sight.trace import Trace
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation_parameters import SimulationParameters


class Simulation(object):
  """Encapsulates start and stop points where a Simulation is active.

  Attributes:
    sight: Reference to the Sight logger via which this simulation is logged.
    label: Human-readable description of this simulation run.
    reference_trace: Describes any reference run of this simulation to which
      this particular run should be compared.
  """

  def __init__(
      self,
      label: str,
      sight: Any,
      parameters: Optional[Dict[Text, Any]],
      reference_trace_file_path: Optional[str] = None,
  ):
    """Creates and enters a simulation block with a given label and parameters.

    Args:
      label: The label that identifies this block.
      sight: The logger instance to which the block is logged.
      parameters: Key-value pairs that identify this block and all of its
        contained objects.
      reference_trace_file_path: Path of the file that contains the Sight log of
        a reference simulation run to compare this run to.
    """
    self.sight = sight
    if sight is None:
      logging.info('<<<Simulation %s', label)
      return

    if not self.sight.is_logging_enabled():
      return
    self.label = label

    # Register this simulation object with Sight.
    sight.widget_simulation_state.simulation = self
    sight.widget_simulation_state.state = {}

    if reference_trace_file_path:
      sight.widget_simulation_state.reference_trace = Trace(
          trace_file_path=reference_trace_file_path)
      sight.widget_simulation_state.reference_trace.advance_to_within_block(
          [sight_pb2.Object.ST_BLOCK_START, sight_pb2.BlockStart.ST_SIMULATION])

    # pytype: disable=attribute-error
    self.sight.enter_block(
        self.label,
        sight_pb2.Object(block_start=sight_pb2.BlockStart(
            sub_type=sight_pb2.BlockStart.ST_SIMULATION)),
        inspect.currentframe().f_back.f_back,
    )
    # pytype: enable=attribute-error
    if parameters:
      with SimulationParameters(parameters, sight):
        pass

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

    # logging.info('>>> %s', self.label)
    if self.sight is None:
      logging.info('>>> %s', self.label)
      return

    # Unregister the associated simulation parameters object with Sight.
    self.sight.widget_simulation_state.simulation_parameters = None

    # Unregister this simulation object with Sight.
    self.sight.widget_simulation_state.simulation = None
    self.sight.widget_simulation_state.state = {}

    # pytype: disable=attribute-error
    self.sight.exit_block(self.label, sight_pb2.Object(),
                          inspect.currentframe().f_back)
    # pytype: enable=attribute-error

  @classmethod
  def run_decision_configuration(
      cls,
      label: str,
      parameters: Optional[Dict[Text, Any]],
      driver_fn: Callable[[Any], Any],
      state_attrs: Dict[str, Tuple[float, float]],
      action_attrs: Dict[str, Tuple[float, float]],
      sight: Any,
      reference_trace_file_path: Optional[str] = None,
  ):
    """Runs this simulation, using the Decision API to configure it.

    Args:
      label: The label that identifies this simulation.
      parameters: Key-value pairs that identify this block and all of its
        contained objects.
      driver_fn: Function that executes a single run of the simulation. This
        function must not carry state across its different invocations, for
        example via files or global variables as it may be called multiple times
        with different configurations.
      state_attrs: Maps the name of each simulation state variable to its
        minimum and maximum possible values.
      action_attrs: Maps the name of each variable that describes possible
        configuration decisions to be made during the simulation's execution to
        its minimum and maximum possible values.
      sight: The logger instance to which the block is logged.
      reference_trace_file_path: Path of the file that contains the Sight log of
        a reference simulation run to compare this run to.
    """

    def run(sight):
      with Simulation(label, sight, parameters, reference_trace_file_path):
        driver_fn(sight)

    decision.run(
        driver_fn=run,
        state_attrs=state_attrs.copy(),
        action_attrs=action_attrs,
        sight=sight,
    )
