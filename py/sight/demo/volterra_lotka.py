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

"""Simulation of the Lotka-Volterra equations using Sight."""

from typing import Dict, Sequence

from absl import app
from absl import flags
from absl import logging

import numpy as np

from sight.proto import sight_pb2
from sight import data_structures
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation import Simulation
from sight.widgets.simulation.simulation_state import SimulationState
from sight.widgets.simulation.simulation_time_step import SimulationTimeStep

_LAST_TS = flags.DEFINE_integer('last_ts', 10,
                                'The final day of the simulation.')
_NUM_ITERS = flags.DEFINE_integer('num_iters', 100,
                                  'The number of steps the solver takes.')

_R0 = flags.DEFINE_integer('r0', 10, 'Initial size of prey population.')
_F0 = flags.DEFINE_integer('f0', 10, 'Initial size of predator population.')
_ALPHA = flags.DEFINE_float('alpha', 1.1, 'Rate of growth of prey population.')
_BETA = flags.DEFINE_float('beta', .4, 'Rate of predator and prey meeting.')
_GAMMA = flags.DEFINE_float('gamma', .4,
                            'Rate of death of predator population.')
_DELTA = flags.DEFINE_float('delta', .1,
                            'Rate of growth of predator population.')


def default_params() -> Dict[str, float]:
  """Returns the run's default configuration parameters.

  These are used if the Decision API doesn't set them to something else
  while searching for a good configuration.
  """
  return {
      'R0': _R0.value,
      'F0': _F0.value,
      'alpha': _ALPHA.value,
      'beta': _BETA.value,
      'gamma': _GAMMA.value,
      'delta': _DELTA.value,
  }


def driver(sight: Sight) -> None:
  """Solves Lotka-Volterra equations using explicit Euler method."""
  steps = np.linspace(0, _LAST_TS.value, _NUM_ITERS.value)

  data_structures.log_var('R', 0, sight)
  data_structures.log_var('F', 0, sight)
  for idx in range(len(steps) - 1):
    with SimulationTimeStep(
        time_step_index=[idx],
        time_step=steps[idx],
        time_step_size=_LAST_TS.value / _NUM_ITERS.value,
        time_step_units=sight_pb2.SimulationTimeStepStart.TSU_UNKNOWN,
        sight=sight):
      action = decision.decision_point('init', sight, default_params)
      logging.info('action=%s', action)
      if idx == 0:
        r = action['R0']
        f = action['F0']
      alpha = action['alpha']
      beta = action['beta']
      gamma = action['gamma']
      delta = action['delta']

      dt = steps[idx + 1] - steps[idx]
      last_r = r
      r = r * (1 + alpha * dt - gamma * dt * f)
      f = f * (1 - beta * dt + delta * dt * last_r)

      # decision.decision_outcome('prey_pop', r - 10 if r > 10 else 0, sight)

      with SimulationState({}, sight):
        data_structures.log_var('R', r, sight)
        data_structures.log_var('F', f, sight)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with Sight(
      sight_pb2.Params(
          label='Volterra-Lotka',
          log_owner='user@domain.com',
          # local=True,
          capacitor_output=True,
          log_dir_path='/tmp/')) as sight:

    Simulation.run_decision_configuration(
        label='Volterra-Lotka',
        parameters={
            'LAST_TS': _LAST_TS.value,
            '_NUM_ITERS': _NUM_ITERS.value,
            'R0': _R0.value,
            'F0': _F0.value,
            'alpha': _ALPHA.value,
            'beta': _BETA.value,
            'gamma': _GAMMA.value,
            'delta': _DELTA.value,
        },
        reference_trace_file_path=flags.FLAGS.reference_run_file,
        driver_fn=driver,
        state_attrs={
            'R': (0, 100),
            'F': (0, 100),
        },
        action_attrs={
            'R0': (0, 20),  # 10
            'F0': (0, 20),  # 10
            'alpha': (0, 2),  # 1.1
            'beta': (0, 1),  # 0.4
            'gamma': (0, 1),  # 0.4
            'delta': (0, 1),  # 0.1
        },
        sight=sight)


if __name__ == '__main__':
  app.run(main)
