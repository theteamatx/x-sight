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
"""Simulation parameters in the Sight log."""

import inspect
from typing import Any, Dict, Text
from helpers.logs.logs_handler import logger as logging

from sight.proto import sight_pb2
from sight import data_structures
from sight.exception import exception


class SimulationParameters(object):
    """Encapsulates log region that documents a simulation's parameters."""

    def __init__(self, parameters: Dict[Text, Any], sight: Any) -> None:
        """Creates and enters a block of a simulation's parameters.

    Args:
      parameters: Key-value pairs that identify this block and all of its
        contained objects.
      sight: The logger instance to which the block is logged.

    Returns:
      The starting location of this simulation parameters block.
    """
        self.parameters = {}
        self.sight = sight
        if sight is None:
            logging.info('<<<SimulationParameters')
            return None

        if not self.sight.is_logging_enabled():
            return None

        # Register this simulation parameters object with Sight.
        sight.widget_simulation_state.simulation_parameters = self

        if sight.widget_simulation_state.reference_trace:
            sight.widget_simulation_state.reference_trace.advance_to_within_block(
                [
                    sight_pb2.Object.ST_BLOCK_START,
                    sight_pb2.BlockStart.ST_SIMULATION_PARAMETERS,
                ])
            sight.widget_simulation_state.reference_trace.collect_current_block(
            )

        # pytype: disable=attribute-error
        self.sight.enter_block(
            'SimulationParameters',
            sight_pb2.Object(block_start=sight_pb2.BlockStart(
                sub_type=sight_pb2.BlockStart.ST_SIMULATION_PARAMETERS)),
            inspect.currentframe().f_back.f_back,
        )
        # pytype: enable=attribute-error

        for key, value in parameters.items():
            # pytype: disable=attribute-error
            data_structures.log_var(key, value, sight,
                                    inspect.currentframe().f_back.f_back)
            # pytype: enable=attribute-error
            self.parameters[key] = value

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
            logging.info('SimulationParameters>>>')
            return

        # Unregister this simulation parameters object with Sight.
        # self.sight.widget_simulation_state.simulation_parameters = None

        # pytype: disable=attribute-error
        self.sight.exit_block(
            'SimulationParameters',
            sight_pb2.Object(block_end=sight_pb2.BlockEnd(
                sub_type=sight_pb2.BlockEnd.ST_SIMULATION_PARAMETERS)),
            inspect.currentframe().f_back,
        )
        # pytype: enable=attribute-error
