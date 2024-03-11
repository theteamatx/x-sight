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

"""Simulation of the Susceptible Infected Recovered model using Sight."""

from typing import Dict, Sequence

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation import Simulation
from sight.widgets.simulation.simulation_state import SimulationState
from sight.widgets.simulation.simulation_time_step import SimulationTimeStep
import os

_LAST_TS = flags.DEFINE_integer(
    'last_ts', 10, 'The final day of the simulation.'
)
_NUM_ITERS = flags.DEFINE_integer(
    'num_iters', 100, 'The number of steps the solver takes.'
)
_NUM_POP = flags.DEFINE_integer(
    'num_pop', 100, 'The number members in the population.'
)
_BETA = flags.DEFINE_float(
    'beta', .1, 'The disease transmission rate.'
)
_GAMMA = flags.DEFINE_float(
    'gamnma', .1, 'The disease recovery rate.'
)


def driver(sight: Sight) -> None:
  """Solves Lotka-Volterra equations using explicit Euler method."""
  I, R = 1, 0
  S = _NUM_POP.value - I - R
  dt = _NUM_ITERS.value / _LAST_TS.value

  # data_structures.log_var('S', S, sight)
  # data_structures.log_var('I', I, sight)
  # data_structures.log_var('R', R, sight)
  action = decision.decision_point('init', sight) 
  print('dt=%s, action=%s' % (dt, action))

  hist = []
  for idx in range(_NUM_ITERS.value - 1):

    dotS = -action['beta'] * S * I / _NUM_POP.value
    dotI = action['beta'] * S * I / _NUM_POP.value - action['gamma'] * I
    dotR = action['gamma'] * I
    print('%d: dotS=%s, dotI=%s, dotR=%s' % (idx, dotS, dotI, dotR))

    S += dotS * dt
    I += dotI * dt
    R += dotR * dt

    # data_structures.log_var('S', S, sight)
    # data_structures.log_var('I', I, sight)
    # data_structures.log_var('R', R, sight)
    hist.append([S, I, R])
  data_structures.log_var('time series',
                      pd.DataFrame(hist, columns=['S', 'I', 'R']),
                      sight)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
      raise app.UsageError('Too many command-line arguments.')

  with Sight(sight_pb2.Params(
          label='SIR',
          bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
      )) as sight:
        decision.run(
            driver_fn=driver,
            description = '''
I am building an SIR model to analyze the progress of Measles infections in Los Angeles during the summer of 2020. 
I need to configure this model's parameters based on data from the Los Angeles County Department of Public Health.
''',
            state_attrs={
            },
            action_attrs={
                'beta': sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0, max_value=1,
                    description='The transmission rate of the disease.'
                ),
                'gamma': sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0, max_value=1,
                    description='The recovery rate of the disease.'
                ),
            },
            sight=sight,
        )


if __name__ == '__main__':
  app.run(main)