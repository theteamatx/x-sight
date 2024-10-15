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
"""Encapsulates the state of the Simulation Sight widget."""

import dataclasses
from typing import Any, Dict, Optional

from sight.trace import Trace
from sight.widgets.simulation.simulation import Simulation
from sight.widgets.simulation.simulation_parameters import SimulationParameters
from sight.widgets.simulation.simulation_state import SimulationState
from sight.widgets.simulation.simulation_time_step import SimulationTimeStep


@dataclasses.dataclass
class SimulationWidgetState:
  """Encapsulates the current running state of a Simulation widget."""

  reference_trace: Optional[Trace]

  simulation: Optional[Simulation]

  simulation_parameters: Optional[SimulationParameters]

  simulation_time_step: Optional[SimulationTimeStep]

  simulation_state: Optional[SimulationState]

  state: Dict[str, Any]

  def __init__(self):
    self.reference_trace = None
    self.simulation = None
    self.simulation_parameters = None
    self.simulation_time_step = None
    self.simulation_state = None
    self.state = {}
