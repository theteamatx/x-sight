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
from helpers.logs.logs_handler import logger as logging
import math
import numpy as np
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation import Simulation
from sight.widgets.simulation.simulation_state import SimulationState
from sight.widgets.simulation.simulation_time_step import SimulationTimeStep
import os

_LAST_TS = flags.DEFINE_integer('last_ts', 10,
                                'The final day of the simulation.')
_NUM_ITERS = flags.DEFINE_integer('num_iters', 100,
                                  'The number of steps the solver takes.')

_R0 = flags.DEFINE_integer('r0', 10, 'Initial size of prey population.')
_F0 = flags.DEFINE_integer('f0', 10, 'Initial size of predator population.')
_ALPHA = flags.DEFINE_float('alpha', 1.1, 'Rate of growth of prey population.')
_BETA = flags.DEFINE_float('beta', 0.4, 'Rate of predator and prey meeting.')
_GAMMA = flags.DEFINE_float('gamma', 0.4,
                            'Rate of death of predator population.')
_DELTA = flags.DEFINE_float('delta', 0.1,
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
  # logging.info('steps=%s', steps)
  data_structures.log_var('R', 0, sight)
  data_structures.log_var('F', 0, sight)
  action = decision.decision_point('init', sight) #, default_params)
  logging.info('action=%s', action)
  print(len(steps))

  with Simulation('Lotka-Volterra', sight, action):
    with SimulationState({}, sight, type=SimulationState.Type.INITIAL):
      for key, val in action.items():
        data_structures.log_var(key, val, sight)

  for idx in range(len(steps) - 1):
    with SimulationTimeStep(
        time_step_index=[idx],
        time_step=steps[idx],
        time_step_size=_LAST_TS.value / _NUM_ITERS.value,
        time_step_units=sight_pb2.SimulationTimeStepStart.TSU_UNKNOWN,
        sight=sight,
    ):
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
      with SimulationState({}, sight, type=SimulationState.Type.DYNAMIC):
        data_structures.log_var('r', r, sight)
        data_structures.log_var('f', f, sight)

    logging.info('r=%s', r)
    if math.isinf(r):
        decision.decision_outcome('prey_pop', -1000, sight)
    else:
        decision.decision_outcome('prey_pop',
                                  r if r < 100 else 100 - 3 * (r - 100), sight)

        # with SimulationState({}, sight):
        #   data_structures.log_var('R', r, sight)
        #   data_structures.log_var('F', f, sight)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    with Sight(
            sight_pb2.Params(
                label='Volterra-Lotka',
                bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
                text_output=True,
            )) as sight:
        # Simulation.run_decision_configuration(
        #     label='Volterra-Lotka',
        #     parameters={
        #         'LAST_TS': _LAST_TS.value,
        #         '_NUM_ITERS': _NUM_ITERS.value,
        #         'R0': _R0.value,
        #         'F0': _F0.value,
        #         'alpha': _ALPHA.value,
        #         'beta': _BETA.value,
        #         'gamma': _GAMMA.value,
        #         'delta': _DELTA.value,
        #     },
        #     reference_trace_file_path=flags.FLAGS.reference_run_file,
        decision.run(
            driver_fn=driver,
            description='''
The Lotka-Volterra equations, also known as the Lotka-Volterra predator-prey model, are a pair of first-order nonlinear differential equations, frequently used to describe the dynamics of biological systems in which two species interact, one as a predator and the other as prey.

The prey are assumed to have an unlimited food supply and to reproduce exponentially, unless subject to predation; this exponential growth is represented in the equation above by the term αx. The rate of predation on the prey is assumed to be proportional to the rate at which the predators and the prey meet; this is represented above by βxy. If either x or y is zero, then there can be no predation. With these two terms the prey equation above can be interpreted as follows: the rate of change of the prey's population is given by its own growth rate minus the rate at which it is preyed upon.

The Lotka-Volterra predator-prey model makes a number of assumptions about the environment and biology of the predator and prey populations:[5]

The prey population finds ample food at all times.
The food supply of the predator population depends entirely on the size of the prey population.
The rate of change of population is proportional to its size.
During the process, the environment does not change in favour of one species, and genetic adaptation is inconsequential.
Predators have limitless appetite.
Both populations can be described by a single variable. This amounts to assuming that the populations do not have a spatial or age distribution that contributes to the dynamics.
''',
            state_attrs={
                'R':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=100,
                    description='The number of prey animals in the population'
                ),
                'F':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=100,
                    description=
                    'The number of predator animals in the population'),
            },
            action_attrs={
                'R0':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description=
                    'The number of predator animals in the population at the start of the simulation.'
                ),
                'F0':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description=
                    'The number of prey animals in the population at the start of the simulation.'
                ),
                'alpha':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description='The growth rate of the prey.',
                ),
                'beta':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description=
                    'The effect of the presence of predators on the prey growth rate, for example by predator eating the prey.'
                ),
                'gamma':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description=
                    'The death rate of the predators independent of the prey.',
                ),
                'delta':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=20,
                    description=
                    'The effect of the presence of prey on the predator\'s growth rate, for example how the predator eating the prey affects the predator population.',
                ),
            },
            sight=sight,
        )


if __name__ == '__main__':
    app.run(main)
