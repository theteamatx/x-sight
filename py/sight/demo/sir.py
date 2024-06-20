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
_MAX_DAYS = flags.DEFINE_integer(
    'max_days', 1000, 'The number of days the solver simulates.'
)
_MAX_POP = flags.DEFINE_integer(
    'max_pop', 10000, 'The number members in the population.'
)
_BETA = flags.DEFINE_float(
    'beta', .1, 'The disease transmission rate.'
)
_GAMMA = flags.DEFINE_float(
    'gamnma', .1, 'The disease recovery rate.'
)


def driver(sight: Sight) -> None:
  """Solves Lotka-Volterra equations using explicit Euler method."""
  dt = .1

  # data_structures.log_var('S', S, sight)
  # data_structures.log_var('I', I, sight)
  # data_structures.log_var('R', R, sight)
  action = decision.decision_point('init', sight) 
  print('dt=%s, action=%s' % (dt, action))
  I, R = 1, 0
  S = int(action['population']) - I - R

  hist = []
  for idx in range(int(int(action['num_days'])/dt) - 1):
    dotS = -action['beta'] * S * I / int(action['population'])
    dotI = action['beta'] * S * I / int(action['population']) - action['gamma'] * I
    dotR = action['gamma'] * I
    

    S += dotS * dt
    I += dotI * dt
    R += dotR * dt

    print('%d: S=(%s/d%s), dotI=(%s/d%s), dotR=(%s/d%s)' % (idx, S, dotS, I, dotI, R, dotR))

    # data_structures.log_var('S', S, sight)
    # data_structures.log_var('I', I, sight)
    # data_structures.log_var('R', R, sight)
    hist.append([S, I, R])
  data_structures.log_var('time series',
                      pd.DataFrame(hist, columns=['S', 'I', 'R']),
                      sight)
  decision.decision_outcome('out', sight, reward=R, outcome={'S': S, 'I': I, 'R': R})


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
              'population': sight_pb2.DecisionConfigurationStart.AttrProps(
                   min_value=0, max_value=_MAX_POP.value,
                   description='The total population of the area affected by the infection.',
                   discrete_prob_dist = sight_pb2.DiscreteProbDist(
                    uniform = sight_pb2.DiscreteProbDist.Uniform(
                        min_val = 0, max_val = _MAX_POP.value))
               ),
               'num_days': sight_pb2.DecisionConfigurationStart.AttrProps(
                   min_value=0, max_value=_MAX_DAYS.value,
                   description='The number of days of the infection being simulated.',
                   discrete_prob_dist = sight_pb2.DiscreteProbDist(
                    uniform = sight_pb2.DiscreteProbDist.Uniform(
                        min_val = 0, max_val = _MAX_DAYS.value))
               ),
               'beta': sight_pb2.DecisionConfigurationStart.AttrProps(
                   min_value=0, max_value=.2,
                   description='The transmission rate of the disease.',
                   continuous_prob_dist = sight_pb2.ContinuousProbDist(
                   uniform = sight_pb2.ContinuousProbDist.Uniform(
                       min_val = 0, max_val = .2))
               ),
               'gamma': sight_pb2.DecisionConfigurationStart.AttrProps(
                   min_value=0, max_value=.2,
                   description='The recovery rate of the disease.',
                   continuous_prob_dist = sight_pb2.ContinuousProbDist(
                   uniform = sight_pb2.ContinuousProbDist.Uniform(
                       min_val = 0, max_val = .2))
               ),
            },
            outcome_attrs={
                'S': sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0, max_value=_MAX_POP.value,
                    description='The number of people who are susceptible to the disease.',
                ),
                'I': sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0, max_value=_MAX_POP.value,
                    description='The number of people who are infected by the disease.',
                ),
                'R': sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0, max_value=_MAX_POP.value,
                    description='The number of people who have recovered from the disease.',
                ),
            },
            sight=sight,
        )


if __name__ == '__main__':
  app.run(main)