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
"""Genetic Algorithms for driving Sight applications."""

from concurrent import futures
import math
import random
from typing import Any, Dict, List, Tuple

from helpers.logs.logs_handler import logger as logging
from overrides import overrides
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.proto import service_pb2


class GeneticAlgorithm(OptimizerInstance):

  def __init__(self):
    super().__init__()
    self.ga_population = []
    self.ga_active_samples = {}
    self.proposals = []
    self.max_population_size = 40
    self.num_decisions = 0
    self.algorithms_tried = {}
    self.algorithms_succeeded_above_min = {}
    self.algorithms_succeeded_best = {}
    self.history = []

  @overrides
  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    response = super(GeneticAlgorithm, self).launch(request)
    response.display_string = 'Genetic Algorithm Launch SUCCESS!'
    logging.info('request.genetic_algorithm_config=%s',
                 request.genetic_algorithm_config)
    # if request.genetic_algorithm_config.max_population_size:
    #   self.max_population_size = max(
    #       3, request.genetic_algorithm_config.max_population_size
    #   )
    ga_config = request.decision_config_params.choice_config[
        request.label].genetic_algorithm_config
    self.max_population_size = ga_config.max_population_size
    return response

  def find_best_worst(
      self, options: List[Dict[str, Any]]) -> Tuple[float, int, float, int]:
    largest_outcome = -math.inf
    largest_idx = -1
    smallest_outcome = math.inf
    smallest_idx = -1
    sum_outcomes = 0
    for i, unit in enumerate(options):
      if unit['outcome'] > largest_outcome:
        largest_outcome = unit['outcome']
        largest_idx = i
      if unit['outcome'] < smallest_outcome:
        smallest_outcome = unit['outcome']
        smallest_idx = i
      sum_outcomes += unit['outcome']

    return (
        largest_outcome,
        largest_idx,
        smallest_outcome,
        smallest_idx,
        sum_outcomes,
    )

  def find_best_worst_probweighted(
      self, options: List[Dict[str, Any]]) -> Tuple[float, int, float, int]:
    (
        largest_outcome,
        largest_idx,
        smallest_outcome,
        smallest_idx,
        sum_outcomes,
    ) = self.find_best_worst(options)
    # logging.info('largest_outcome=%s, largest_idx=%s, smallest_outcome=%s, smallest_idx=%s, sum_outcomes=%s', largest_outcome, largest_idx, smallest_outcome, smallest_idx, sum_outcomes)

    sum_of_max_adjusted_outcomes = largest_outcome * len(options) - sum_outcomes
    smallest_outcome_choice = random.uniform(0, sum_of_max_adjusted_outcomes)
    logging.info(
        'sum_of_max_adjusted_outcomes=%s, smallest_outcome_choice=%s',
        sum_of_max_adjusted_outcomes,
        smallest_outcome_choice,
    )

    cumulative_outcomes_sum = 0
    smallest_outcome = math.inf
    smallest_idx = -1
    for i, unit in enumerate(options):
      cumulative_outcomes_sum += largest_outcome - unit['outcome']
      # logging.info('unit[outcome]=%s, cumulative_outcomes_sum=%s, found=%s', unit['outcome'], cumulative_outcomes_sum, smallest_outcome_choice < cumulative_outcomes_sum)
      if smallest_outcome_choice <= cumulative_outcomes_sum:
        return largest_outcome, largest_idx, unit['outcome'], i

    logging.error(
        ('WARNING: smallest_outcome_choice=%s,'
         ' sum_of_max_adjusted_outcomes=%s but we failed to find the index'
         ' of this unit'),
        smallest_outcome_choice,
        sum_of_max_adjusted_outcomes,
    )
    return largest_outcome, largest_idx, smallest_outcome, smallest_idx

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    logging.info('%s| ga_population(#%d)=', request.worker_id,
                 len(self.ga_population))
    for member in sorted(self.ga_population,
                         key=lambda p: p['outcome'],
                         reverse=True):
      logging.info('%s| %s: %s', request.worker_id, member['outcome'],
                   member['action'])

    self.num_decisions += 1
    if (len(self.ga_population) < self.max_population_size or
        random.randint(1, 100) <= 5):
      algorithm = 'random_sample'
      # Randomly sample an action.
      next_action = {}
      for key in self.actions.keys():
        next_action[key] = random.uniform(self.actions[key].min_value,
                                          self.actions[key].max_value)
        # logging.info("   [%s - %s]: %s", self.actions[key].min_value,
        #           self.actions[key].max_value,
        #           next_action[key])

      if len(self.ga_population) >= self.max_population_size:
        largest_outcome, largest_idx, smallest_outcome, smallest_idx = (
            self.find_best_worst_probweighted(self.ga_population))
        # Remove the worst member of the population
        del self.ga_population[smallest_idx]

      logging.info(
          '%s| Randomly sample: next_action  : %s',
          request.worker_id,
          next_action,
      )
    else:
      largest_outcome, largest_idx, smallest_outcome, smallest_idx = (
          self.find_best_worst_probweighted(self.ga_population))

      # logging.info('Retrying largest=%s', self.ga_population[spouse_idx])
      # next_action = dict(self.ga_population[largest_idx]['action'])
      # # Remove the chosen member of the population
      # logging.info('deleting largest unit=%s', self.ga_population[largest_idx])
      # del self.ga_population[largest_idx]

      if self.proposals and random.randint(0, 10) < 5:
        (
            prop_largest_outcome,
            prop_largest_idx,
            prop_smallest_outcome,
            prop_smallest_idx,
        ) = self.find_best_worst_probweighted(self.proposals)
        logging.info(
            '%s| Best proposal: %s: %s',
            request.worker_id,
            self.proposals[prop_largest_idx]['outcome'],
            self.proposals[prop_largest_idx]['action'],
        )
        next_action = self.proposals[prop_largest_idx]['action']
        algorithm = 'best_proposal'
        del self.proposals[prop_largest_idx]
      else:
        spouse_idx = random.randint(0, len(self.ga_population) - 1)
        # logging.info('smallest_idx=%s, largest_idx=%s, spouse_idx=%s',
        #              smallest_idx, largest_idx, spouse_idx)
        while spouse_idx == smallest_idx or spouse_idx == largest_idx:
          spouse_idx = (spouse_idx + 1) % len(self.ga_population)
        logging.info(
            '%s| smallest_idx=%s, largest_idx=%s, spouse_idx=%s',
            request.worker_id,
            smallest_idx,
            largest_idx,
            spouse_idx,
        )

        if random.randint(0, 9) > 4:
          # Mate largest_idx and spouse_idx
          logging.info(
              '%s| Mating largest unit=%s : %s',
              request.worker_id,
              self.ga_population[largest_idx]['outcome'],
              self.ga_population[largest_idx]['action'],
          )
          logging.info(
              '%s|    and spouse=%s : %s',
              request.worker_id,
              self.ga_population[spouse_idx]['outcome'],
              self.ga_population[spouse_idx]['action'],
          )
          next_action = {}
          keys = sorted(self.actions.keys())
          cross_idx = random.randint(0, len(keys) - 1)
          logging.info('%s|    at cross_idx=%d', request.worker_id, cross_idx)
          for i, key in enumerate(keys):
            if i < cross_idx:
              next_action[key] = self.ga_population[spouse_idx]['action'][key]
            else:
              next_action[key] = self.ga_population[largest_idx]['action'][key]
          algorithm = 'mating'
        else:
          mutation_prob = random.randint(0, 100)
          logging.info(
              '%s| mutating mutation_prob=%s, spouse=%s: %s',
              request.worker_id,
              mutation_prob,
              self.ga_population[spouse_idx]['outcome'],
              self.ga_population[spouse_idx]['action'],
          )
          next_action = {}
          for key in self.actions.keys():
            if random.randint(0, 999) <= mutation_prob:
              next_action[key] = random.uniform(self.actions[key].min_value,
                                                self.actions[key].max_value)
              # next_action[key] = self.ga_population[spouse_idx]['action'][key] * random.uniform(.9, 1.1)
              # if next_action[key] < self.actions[key].min_value:
              #   next_action[key] = self.actions[key].min_value
              # elif next_action[key] > self.actions[key].max_value:
              #   next_action[key] = self.actions[key].max_value
            else:
              next_action[key] = self.ga_population[spouse_idx]['action'][key]
            # logging.info('received_action[%s]=%s original=%s', key, next_action[key], claim_year_sold_delay)
          algorithm = f'mutating_{mutation_prob}'
        logging.info('%s| new next_action=%s', request.worker_id, next_action)

      # Remove the worst member of the population
      # logging.info('deleting smallest unit=%s', self.ga_population[smallest_idx])
      del self.ga_population[smallest_idx]

    self.ga_active_samples[request.worker_id] = {
        'action': next_action,
        'algorithm': algorithm,
    }

    dp_response = service_pb2.DecisionPointResponse()
    dp_response.action.extend(param_dict_to_proto(next_action))
    dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    algorithm = self.ga_active_samples[request.worker_id]['algorithm']
    if algorithm not in self.algorithms_tried:
      self.algorithms_tried[algorithm] = 0
      self.algorithms_succeeded_above_min[algorithm] = 0
      self.algorithms_succeeded_best[algorithm] = 0
    self.algorithms_tried[algorithm] += 1

    if self.ga_population:
      largest_outcome, largest_idx, smallest_outcome, smallest_idx = (
          self.find_best_worst_probweighted(self.ga_population))
      if request.decision_outcome.reward >= smallest_outcome:
        self.algorithms_succeeded_above_min[algorithm] += 1
      if request.decision_outcome.reward >= largest_outcome:
        self.algorithms_succeeded_best[algorithm] += 1

    self.ga_population.append({
        'outcome': request.decision_outcome.reward,
        'action': self.ga_active_samples[request.worker_id]['action'],
    })
    self.history.append({
        'algorithm': algorithm,
        'outcome': request.decision_outcome.reward,
        'action': self.ga_active_samples[request.worker_id]['action'],
        'worker_id': request.worker_id,
    })
    logging.info(
        '%s| FinalizeEpisode member=%s: %s / %s',
        request.worker_id,
        request.decision_outcome.reward,
        self.ga_active_samples[request.worker_id]['algorithm'],
        self.ga_active_samples[request.worker_id]['action'],
    )
    del self.ga_active_samples[request.worker_id]
    logging.info(
        '%s| FinalizeEpisode #ga_active_samples=%s',
        request.worker_id,
        len(self.ga_active_samples),
    )
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    response = (
        f'[GeneticAlgorithm (max_population_size={self.max_population_size},'
        f' num_decisions={self.num_decisions}):\n')
    response += f'  ga_population(#{len(self.ga_population)}):\n'
    keys = sorted(self.actions.keys())
    response += '  idx,outcome,' + ','.join(keys) + '\n'
    for i, unit in enumerate(
        sorted(self.ga_population, key=lambda p: p['outcome'], reverse=True)):
      response += (f'  {i},{unit["outcome"]:.5F},' +
                   ','.join([str(unit['action'][key]) for key in keys]) + '\n')

    response += f'  ga_active_samples(#{len(self.ga_active_samples)}):\n'
    response += '  worker_id,algorithm,' + ','.join(keys) + '\n'
    for worker_id, sample in self.ga_active_samples.items():
      response += (f'  {worker_id},{sample["algorithm"]},' +
                   ','.join([str(sample['action'][key]) for key in keys]) +
                   '\n')
    response += ']'

    response += f'  proposals(#{len(self.proposals)}):\n'
    response += '  idx,outcome,' + ','.join(keys) + '\n'
    for i, unit in enumerate(
        sorted(self.proposals, key=lambda p: p['outcome'], reverse=True)):
      response += (f'  {i},{unit["outcome"]:.5F},' +
                   ','.join([str(unit['action'][key]) for key in keys]) + '\n')
      if i > 50:
        break

    response += f'  algorithms:\n'
    for algorithm in sorted(self.algorithms_tried.keys()):
      response += ('    %s: tried=%s, algorithms_succeeded_above_min=%.4E,'
                   ' algorithms_succeeded_best=%.4E\n' % (
                       algorithm,
                       self.algorithms_tried[algorithm],
                       self.algorithms_succeeded_above_min[algorithm] /
                       self.algorithms_tried[algorithm],
                       self.algorithms_succeeded_best[algorithm] /
                       self.algorithms_tried[algorithm],
                   ))

    response += '  history:\n'
    for i, h in enumerate(self.history):
      response += '   %d: %s\n' % (i, h)

    return service_pb2.CurrentStatusResponse(response_str=response)

  def propose_action(
      self, request: service_pb2.ProposeActionRequest
  ) -> service_pb2.ProposeActionResponse:
    action = {}
    for key, value in request.action.items():
      action[key] = value

    largest_outcome, largest_idx, smallest_outcome, smallest_idx = (
        self.find_best_worst_probweighted(self.ga_population))
    if request.outcome.reward >= smallest_outcome:
      self.proposals.append({
          'action': action,
          'outcome': request.outcome.reward,
      })
      logging.info(
          '%s| Accepted Proposal %s: %s',
          request.worker_id,
          request.outcome.reward,
          action,
      )
    else:
      logging.info(
          '%s| Rejected Proposal %s: %s',
          request.worker_id,
          request.outcome.reward,
          action,
      )
    return service_pb2.ProposeActionResponse()
