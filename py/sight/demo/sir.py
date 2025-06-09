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

import os
from typing import Dict, Sequence

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
import pandas as pd
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.simulation.simulation import Simulation
from sight.widgets.simulation.simulation_state import SimulationState
from sight.widgets.simulation.simulation_time_step import SimulationTimeStep

_LAST_TS = flags.DEFINE_integer('last_ts', 10,
                                'The final day of the simulation.')
_MAX_DAYS = flags.DEFINE_integer('max_days', 1000,
                                 'The number of days the solver simulates.')
_MAX_POP = flags.DEFINE_integer('max_pop', 10000,
                                'The number members in the population.')
_BETA = flags.DEFINE_float('beta', .1, 'The disease transmission rate.')
_GAMMA = flags.DEFINE_float('gamnma', .1, 'The disease recovery rate.')


def driver(sight: Sight) -> None:
  """Solves Lotka-Volterra equations using explicit Euler method."""
  dt = .1

  # data_structures.log_var('S', S, sight)
  # data_structures.log_var('I', I, sight)
  # data_structures.log_var('R', R, sight)
  action = decision.decision_point('init', sight)
  print('dt=%s, action=%s' % (dt, action))
  I = int(action['I0'])
  R = 0
  S = int(action['population']) - I - R

  hist = []
  for idx in range(int(int(action['num_days']) / dt) - 1):
    dotS = -action['beta'] * S * I / int(action['population'])
    dotI = action['beta'] * S * I / int(
        action['population']) - action['gamma'] * I
    dotR = action['gamma'] * I

    S += dotS * dt
    I += dotI * dt
    R += dotR * dt

    print('%d: S=(%s/d%s), dotI=(%s/d%s), dotR=(%s/d%s)' %
          (idx, S, dotS, I, dotI, R, dotR))

    # data_structures.log_var('S', S, sight)
    # data_structures.log_var('I', I, sight)
    # data_structures.log_var('R', R, sight)
    hist.append([S, I, R])
  data_structures.log_var('time series',
                          pd.DataFrame(hist, columns=['S', 'I', 'R']), sight)
  decision.decision_outcome('out',
                            sight,
                            reward=R,
                            outcome={
                                'S': S,
                                'I': I,
                                'R': R
                            })


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with Sight(
      sight_pb2.Params(
          label='SIR',
          bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
      )) as sight:
    decision.run(
        driver_fn=driver,
        description='''
The SIR model is one of the simplest compartmental models, and many models are derivatives of this basic form. The model consists of three compartments:

S: The number of susceptible individuals. When a susceptible and an infectious individual come into "infectious contact", the susceptible individual contracts the disease and transitions to the infectious compartment.
I: The number of infectious individuals. These are individuals who have been infected and are capable of infecting susceptible individuals.
R for the number of removed (and immune) or deceased individuals. These are individuals who have been infected and have either recovered from the disease and entered the removed compartment, or died. It is assumed that the number of deaths is negligible with respect to the total population. This compartment may also be called "recovered" or "resistant".
This model is reasonably predictive[11] for infectious diseases that are transmitted from human to human, and where recovery confers lasting resistance, such as measles, mumps, and rubella.

These variables (S, I, and R) represent the number of people in each compartment at a particular time. To represent that the number of susceptible, infectious, and removed individuals may vary over time (even if the total population size remains constant), we make the precise numbers a function of t (time): S(t), I(t), and R(t). For a specific disease in a specific population, these functions may be worked out in order to predict possible outbreaks and bring them under control.[11] Note that in the SIR model, R(0) and R_{0}} are different quantities – the former describes the number of recovered at t = 0 whereas the latter describes the ratio between the frequency of contacts to the frequency of recovery.

As implied by the variable function of t, the model is dynamic in that the numbers in each compartment may fluctuate over time. The importance of this dynamic aspect is most obvious in an endemic disease with a short infectious period, such as measles in the UK prior to the introduction of a vaccine in 1968. Such diseases tend to occur in cycles of outbreaks due to the variation in number of susceptibles (S(t)) over time. During an epidemic, the number of susceptible individuals falls rapidly as more of them are infected and thus enter the infectious and removed compartments. The disease cannot break out again until the number of susceptibles has built back up, e.g. as a result of offspring being born into the susceptible compartment.[citation needed]

Each member of the population typically progresses from susceptible to infectious to recovered. This can be shown as a flow diagram in which the boxes represent the different compartments and the arrows the transition between compartments.
''',
        state_attrs={},
        action_attrs={
            'population':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=_MAX_POP.value,
                    description=
                    'The total population of the area affected by the infection.',
                    discrete_prob_dist=sight_pb2.DiscreteProbDist(
                        uniform=sight_pb2.DiscreteProbDist.Uniform(
                            min_val=0, max_val=_MAX_POP.value))),
            'num_days':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=_MAX_DAYS.value,
                    description=
                    'The number of days of the infection being simulated.',
                    discrete_prob_dist=sight_pb2.DiscreteProbDist(
                        uniform=sight_pb2.DiscreteProbDist.Uniform(
                            min_val=0, max_val=_MAX_DAYS.value))),
            'beta':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=.2,
                    description='The transmission rate of the disease.',
                    continuous_prob_dist=sight_pb2.ContinuousProbDist(
                        uniform=sight_pb2.ContinuousProbDist.Uniform(
                            min_val=0, max_val=.2))),
            'gamma':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=.2,
                    description='The recovery rate of the disease.',
                    continuous_prob_dist=sight_pb2.ContinuousProbDist(
                        uniform=sight_pb2.ContinuousProbDist.Uniform(
                            min_val=0, max_val=.2))),
            'I0':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=1000,
                    description=
                    'The number of individuals infected at the start of the epidemic.',
                    discrete_prob_dist=sight_pb2.DiscreteProbDist(
                        uniform=sight_pb2.DiscreteProbDist.Uniform(
                            min_val=0, max_val=1000))),
        },
        outcome_attrs={
            'S':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=_MAX_POP.value,
                    description=
                    'The number of people who are susceptible to the disease.',
                ),
            'I':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=_MAX_POP.value,
                    description=
                    'The number of people who are infected by the disease.',
                ),
            'R':
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0,
                    max_value=_MAX_POP.value,
                    description=
                    'The number of people who have recovered from the disease.',
                ),
        },
        sight=sight,
    )


if __name__ == '__main__':
  app.run(main)
