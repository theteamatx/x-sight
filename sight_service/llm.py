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
"""LLM-based optimization for driving Sight applications."""

from concurrent import futures
import json
import os
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import google.auth.transport.requests
import google.generativeai as genai
from helpers.logs.logs_handler import logger as logging
from overrides import overrides
import requests
from sight.proto import sight_pb2
from sight.utils.proto_conversion import convert_dict_to_proto
from sight.utils.proto_conversion import convert_proto_to_dict
from sight_service.bayesian_opt import BayesianOpt
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2

# _GENAI_API_KEY = os.environ['GENAI_API_KEY']


class LLM(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.

  Attributes:
    script: The script of the conversation accrued so far.
  """

  def __init__(self):
    super().__init__()
    # genai.configure(api_key=_GENAI_API_KEY)
    genai.configure(api_key="_GENAI_API_KEY")
    self._intro = ''
    self._history = []
    self._actions_to_do = []
    self._history_len_for_prompt = 20
    self._num_decision_points = 0
    # self.last_outcome = None
    self._lock = threading.RLock()
    self._waiting_on_tell = False
    self._response_ready = False
    self._response_for_listen = ''
    self._waiting_on_llm_response = False

  def _attr_summary(
      self, key: str,
      attr: sight_pb2.DecisionConfigurationStart.AttrProps) -> str:
    """Returns a summary of an attribute for the LLM."""
    if attr.min_value < attr.max_value:
      return (f'"{key}": {{ "description": {attr.description}, "min_value":'
              f' {attr.min_value}, "max_value": {attr.max_value} }},')
    return f'"{key}": {{ "description": {attr.description} }},'

  @overrides
  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    response = super(LLM, self).launch(request)
    logging.info('LLM request=%s', request)
    self._llm_config = request.decision_config_params.choice_config[
        request.label].llm_config
    logging.info('LLM config=%s', self._llm_config)
    self._bayesian_opt = BayesianOpt()
    self._bayesian_opt.launch(request)

    self._intro += ''
    if (self._llm_config.goal ==
        sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_OPTIMIZE):
      self._intro = (
          'You are controlling an agent that is trying to reach a goal. The'
          ' agent is described as follows.\n')
    self._intro += f'"{self._llm_config.description}"\n'
    if (self._llm_config.goal ==
        sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_OPTIMIZE):
      self._intro += (
          'The simulation will periodically report its state and then ask you '
          + 'to select an action for it to perform. After it has performed'
          ' this ' +
          'action it will report back the numeric outcome of the this'
          ' action. ' +
          'Higher outcome values are better than low outcome values. Your'
          ' job ' + 'is to choose actions that maximize the outcome values.\n')
    if len(self.state) > 0:
      self._intro += (
          'The state of the simulation consists of the following attributes: \n'
      )
      self._intro += ('  {\n    ' + '\n    '.join(
          [self._attr_summary(key, p) for key, p in self.state.items()]) +
                      '}\n')
    self._intro += 'The possible actions you need to select are: \n'
    self._intro += ('  {\n    ' + '\n    '.join(
        [self._attr_summary(key, p) for key, p in self.actions.items()]) +
                    '}\n')
    self._intro += 'The possible outcomes you will observe are: \n'
    self._intro += ('  {\n    ' + '\n    '.join(
        [self._attr_summary(key, p) for key, p in self.outcomes.items()]) +
                    '}\n')
    self._intro += '========================\n'

    logging.info(
        'INTERACTIVE=%s',
        self._llm_config.goal ==
        sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE,
    )
    if (self._llm_config.goal ==
        sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE):
      self._waiting_on_tell = True
      self._response_ready = False
      self._response_for_listen = ''
      self._waiting_on_llm_response = False
    else:
      detail_prompt = (
          'Please summarize everything you know about these parameters for the'
          ' above application area, detail the steps that need to be taken to'
          ' create a good estimate these parameters.\n')
      self._intro += (detail_prompt + self._ask(self._intro + detail_prompt) +
                      '\n')

      detail_prompt = (
          'Based on this plan describe the most reasonable estimate of these'
          ' parameters\n')
      self._intro += (detail_prompt + self._ask(self._intro + detail_prompt) +
                      '\n')

    response.display_string = 'LLM SUCCESS! ' + self._intro
    logging.info('self._intro=%s', self._intro)
    return response

  def _random_state(self) -> Dict[str, float]:
    """Returns a random state."""
    s = {}
    for key, p in self.state.items():
      s[key] = (p.max_value - p.min_value) * random.random() + p.min_value
    return s

  def _random_action(self) -> Dict[str, float]:
    """Returns a random action."""
    a = {}
    for key, p in self.actions.items():
      a[key] = (p.max_value - p.min_value) * random.random() + p.min_value
    return a

  def _random_outcome(self) -> Dict[str, float]:
    """Returns a random outcome."""
    o = {}
    for key, p in self.outcomes.items():
      o[key] = (p.max_value - p.min_value) * random.random() + p.min_value
    return o

  def _random_event(self) -> Dict[str, Any]:
    return {
        'state': self._random_state(),
        'action': self._random_action(),
        'outcome': self._random_outcome(),
        # random.random(),
    }

  def _filtered_history(self, include_example_action: bool) -> List[Any]:
    ordered_history = self._history[:-1].copy()
    # logging.info(
    #     '#hist=%d ordered_history[#%d]=%s',
    #     len(self._history),
    #     len(ordered_history),
    #     ordered_history,
    # )
    ordered_history = sorted(
        ordered_history,
        key=lambda h: -h['outcome']
        if 'outcome' in h and isinstance(h['outcome'], float) else 0,
    )
    if len(ordered_history) > self._history_len_for_prompt:
      ordered_history = ordered_history[0:self._history_len_for_prompt - 1]
    random.shuffle(ordered_history)

    # If this is the first question, add a random event to serve as an example
    # of the format.
    # if include_example_action and len(ordered_history) == 0:
    #   ordered_history.append(self._random_event())

    # logging.info(
    #     'ordered_history[#%d]=%s',
    #     len(ordered_history),
    #     ordered_history,
    # )
    # if worker_id is None:
    if len(self._history) == 0:
      return ordered_history
    return ordered_history + [self._history[-1]]

  def _hist_event_to_text(self, event: Dict, last_outcome: float,
                          is_last_event: bool) -> Tuple[str, Any]:
    t = ''
    if len(event['state']) > 0:
      t += 'Decision State:\n'
      t += ('    {' +
            ', '.join([f'"{k}": {v}' for k, v in event['state'].items()]) +
            '}\n')
    # t += 'Decision Action (json format): '
    if event['action'] is not None or is_last_event:
      t += 'Simulation parameters (json format): '
    if event['action'] is not None:
      t += ('    {' + ', '.join(
          [f'"{key}": {value}' for key, value in event['action'].items()]) +
            '}\n')
    if event['outcome'] is not None:
      # t += 'Decision Outcome: ' + str(event['outcome']) + '\n'
      t += 'Simulation Outcome (json format): ' + str(event['outcome']) + '\n'
      if (self._llm_config.goal != sight_pb2.DecisionConfigurationStart.
          LLMConfig.LLMGoal.LM_INTERACTIVE):
        if last_outcome is not None:
          if last_outcome < event['outcome'] - 0.1:
            t += '  This is a better outcome than the last time.\n'
          elif last_outcome > event['outcome'] + 0.1:
            t += '  This is a worse outcome than the last time.\n'
          else:
            t += '  This is a similar outcome to the last time.\n'
      t += '========================\n'
      last_outcome = event['outcome']
    return t, last_outcome

  def _history_to_text(self, include_example_action: bool = True) -> str:
    t = ''
    last_outcome = None
    hist = self._filtered_history(include_example_action)
    # logging.info(
    #     '_history_to_text() include_example_action=%s hist=%s',
    #     include_example_action,
    #     hist,
    # )
    # if include_example_action and (
    #     len(hist) == 0 or (len(hist) == 1 and hist[0]['outcome'] is None)
    # ):
    #   logging.info('_history_to_text() Adding random_event')
    # t += self._hist_event_to_text(self._random_event(), None, False)
    for i, event in enumerate(hist):
      # logging.info('_history_to_text event=%s', event)
      event_text, last_outcome = self._hist_event_to_text(
          event, last_outcome, i == len(hist) - 1)
      t += event_text
    return t

  def _history_to_chat(
      self,
      worker_id: str,
      include_example_action: bool = True) -> List[Dict[str, str]]:
    chat = []
    last_outcome = None
    last_outcome_message = ''
    for h in self._filtered_history(include_example_action):
      if len(h['state']) > 0:
        chat.append({
            'author': 'USER',
            'content':
                (last_outcome_message + 'Decision State:\n' + '    {' +
                 ', '.join([f'"{k}": {v}' for k, v in h['state'].items()]) +
                 '}\n' + 'Please provide the Decision Action (json format):\n'),
        })
      if h['action'] is not None:
        chat.append({
            'author': 'AI',
            'content': (+ 'Decision Action:\n' + '    {{' + ', '.join(
                [f'"{key}": {value}' for key, value in h['action'].items()]) +
                        '}'),
        })
      if h['outcome'] is not None:
        last_outcome_message = 'Decision Outcome: ' + str(h['outcome']) + '\n'
        if (self._llm_config.goal != sight_pb2.DecisionConfigurationStart.
            LLMConfig.LLMGoal.LM_INTERACTIVE):
          if last_outcome is not None:
            if last_outcome < h['outcome'] - 0.1:
              last_outcome_message += (
                  '  This is a better outcome than the last time.\n')
            elif last_outcome > h['outcome'] + 0.1:
              last_outcome_message += (
                  '  This is a worse outcome than the last time.\n')
            else:
              last_outcome_message += (
                  '  This is a similar outcome to the last time.\n')
    return chat

  # def _params_to_dict(self, dp: sight_pb2.DecisionParam) -> Dict[str, float]:
  #   """Returns the dict representation of a DecisionParams proto"""
  #   d = {}
  #   logging.info('params_to_dict() dp.params=%s', dp.params)
  #   for a in dp.params:
  #     logging.info('params_to_dict()     a=%s', a)
  #     d[a.key] = a.value.double_value
  #   return d

  def _get_creds(self) -> Any:
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds

  def _get_req_headers(self) -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {self._get_creds().token}',
        'Content-Type': 'application/json; charset=utf-8',
    }

  def _ask(self, prompt) -> str:
    if (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
        LLMConfig.LLMAlgorithm.LA_TEXT_BISON):
      return self._ask_text_bison(prompt)
    elif (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
          LLMConfig.LLMAlgorithm.LA_CHAT_BISON):
      return self._ask_chat_bison(prompt)
    elif (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
          LLMConfig.LLMAlgorithm.LA_GEMINI_PRO):
      return self._ask_gemini_pro(prompt)
    else:
      raise ValueError(f'Invalid algorithm {self._llm_config.algorithm}')

  def _ask_text_bison(self, prompt) -> str:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/text-bison-32k:predict",
          data=json.dumps({
              'instances': [{
                  'prompt': prompt
              }],
              'parameters': {
                  'temperature': 0.2,
                  'maxOutputTokens': 2048,
                  'topK': 40,
                  'topP': 0.55,
                  # "groundingConfig": string,
                  # "stopSequences": [ string ],
                  # "candidateCount": integer,
                  # "logprobs": integer,
                  # "presencePenalty": float,
                  # "frequencyPenalty": float,
                  # "logitBias": map<string, float>,
                  'echo': False,
              },
          }),
          headers=self._get_req_headers(),
      ).json()
      # logging.info('response=%s', response)
      if ('error' in response or
          response['predictions'][0]['content'].strip() == ''):
        continue
      return response['predictions'][0]['content'].strip()

  def _get_action(self, worker_id: str) -> List[Dict[str, float]]:
    if (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
        LLMConfig.LLMAlgorithm.LA_TEXT_BISON):
      return self._action_from_text_bison(worker_id)
    elif (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
          LLMConfig.LLMAlgorithm.LA_CHAT_BISON):
      return self._action_from_chat_bison(worker_id)
    elif (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
          LLMConfig.LLMAlgorithm.LA_GEMINI_PRO):
      return self._action_from_gemini_pro(worker_id)
    else:
      raise ValueError(f'Invalid algorithm {self._llm_config.algorithm}')

  def _action_from_text_bison(self, worker_id: str) -> List[Dict[str, float]]:
    logging.info('ask_text_bison')
    logging.info(self._intro + '\n' + self._history_to_text())
    while True:
      text = self._ask_text_bison(self._intro + '\n' + self._history_to_text())
      logging.info('text=[%s]', text)
      # text = text.removeprefix('```json\n')
      # logging.info('text=[%s]', text)
      text = text.strip('`').split('\n')[0]
      # text = text.split('\n')[0].strip()
      logging.info('text=[%s]', text)
      try:
        return [json.loads(text)]
      except json.decoder.JSONDecodeError:
        continue

  def _ask_chat_bison(self, prompt, message) -> str:
    response = requests.post(
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/chat-bison-32k:predict",
        data=json.dumps({
            'instances': [{
                'context': prompt,
                'messages': message,
            },],
            'parameters': {
                'temperature': 0.2,
                'maxOutputTokens': 2048,
                'topK': 40,
                'topP': 0.55,
                # "groundingConfig": string,
                # "stopSequences": [ string ],
                # "candidateCount": integer,
                # "logprobs": integer,
                # "presencePenalty": float,
                # "frequencyPenalty": float,
                # "logitBias": map<string, float>,
                'echo': False,
            },
        }),
        headers=self._get_req_headers(),
    ).json()
    # logging.info('response=%s', response)
    # logging.info(
    #     "response['predictions']=%s", response['predictions'][0]['candidates']
    # )
    # if 'error' in response or response['predictions'][0]['content'].strip() == '':
    #  continue
    return response['predictions'][0]['candidates'][0]['content'].strip()

  def _action_from_chat_bison(self, worker_id: str) -> List[Dict[str, float]]:
    while True:
      text = self._ask_chat_bison(self._intro, self._history_to_chat(worker_id))
      logging.info('text=[%s]', text)
      try:
        return [json.loads(text)]
      except json.decoder.JSONDecodeError:
        continue

  def _ask_gemini_pro(self, prompt) -> str:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent",
          data=json.dumps({
              'contents': {
                  'role': 'user',
                  'parts': {
                      'text': prompt
                  },
              },
              'safety_settings': {
                  'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                  'threshold': 'BLOCK_LOW_AND_ABOVE',
              },
              'generation_config': {
                  'temperature': 0.9,
                  'topP': 1,
                  'topK': 1,
                  'maxOutputTokens': 8192,
                  # "stopSequences": [".", "?", "!"]
              },
          }),
          headers=self._get_req_headers(),
      ).json()
      # logging.info('response=%s', response)
      if len(response) == 0:
        continue
      text = ''
      for r in response:
        if 'content' in r['candidates'][0] and 'parts' in r['candidates'][0]['content']:
          text += r['candidates'][0]['content']['parts'][0]['text']
      text = text.strip()
      if text == '':
        continue
      return text

  def _action_from_gemini_pro(self, worker_id: str) -> List[Dict[str, float]]:
    while True:
      logging.info('ask_geminipro')
      prompt = self._intro + '\n'
      random_sample, _ = self._hist_event_to_text(self._random_event(), None,
                                                  False)
      prompt += 'Example request: ' + random_sample + '\n'
      prompt += self._history_to_text()
      logging.info('prompt=%s', prompt)
      text = self._ask_gemini_pro(prompt)
      if text.startswith('```json'):
        text = [text.removeprefix('```json').removesuffix('```')]
      else:
        text = text.split('\n')
      logging.info('text=[%s]', text)

      actions = []
      for i in range(0, len(text), 3):
        try:
          logging.info('%d: processed %s', i, text[i])
          action = text[i].removeprefix('Simulation parameters (json format):')
          logging.info('%d: action=%s', i, action)
          actions.append(json.loads(action))
        except json.decoder.JSONDecodeError:
          continue
      if len(actions) == 0:
        continue
      return actions

  def _is_done(self, worker_id: str) -> Tuple[bool, str]:
    """Checks with the LLM to see whether it has enough information to answer.

    Returns a tuple with a boolean that indicates whether the question can
    be answered and if so, the answer string.
    """
    if (self._llm_config.algorithm == sight_pb2.DecisionConfigurationStart.
        LLMConfig.LLMAlgorithm.LA_GEMINI_PRO):
      return self._is_done_from_gemini_pro(worker_id)
    return False, ''

  def _is_done_from_gemini_pro(self, worker_id: str) -> Tuple[bool, str]:
    question = (
        self._intro + '\n' + self._history_to_text(False) +
        '\nHas the question been fully answered, including all of its'
        ' clauses? Answer Y if yes or N if there are any additional'
        ' simulations that need to be performed to fully answer the question.')
    logging.info('_is_done_from_gemini_pro question=%s', question)
    text = self._ask_gemini_pro(question)
    logging.info('_is_done_from_gemini_pro text=%s', text)
    if not text.lower().startswith('y'):
      logging.info('_is_done_from_gemini_pro NOT DONE')
      return False, ''
    question = (self._intro + '\n' + self._history_to_text(False) +
                "\nWhat is the answer to the user's question?")
    text = self._ask_gemini_pro(question)
    logging.info('_is_done_from_gemini_pro answer=%s', text)
    return True, text

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    logging.info('DecisionPoint request=%s', request)
    # self._append_outcome(request.decision_outcome.reward)
    self._lock.acquire()

    dp_response = service_pb2.DecisionPointResponse()

    if (self._llm_config.goal
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE
        and self._waiting_on_tell):
      self._lock.release()
      dp_response.action_type = (
          service_pb2.DecisionPointResponse.ActionType.AT_RETRY)
      return dp_response

    if len(self._history) > 0 and 'outcome' not in self._history[0]:
      if len(request.decision_outcome.outcome_params) > 0:
        self._history[-1]['outcome'] = convert_proto_to_dict(
            request.decision_point.outcome_params)
      else:
        self._history[-1]['outcome'] = request.decision_outcome.reward
      # self.last_outcome = self._history[-1]['outcome']
    # self.script += 'Decision State:\n'
    # self.script += '    {' + ', '.join([
    #     f'"{p.key}": {p.value.double_value}'
    #       for p in request.decision_point.state_params
    #     ]) + '}\n'
    # self.script += 'Decision Action (json format):\n'
    self._history.append({
        'state': convert_proto_to_dict(request.decision_point.state_params),
        'action': None,
        'outcome': None,
    })

    if self._actions_to_do:
      selected_actions = [self._actions_to_do.pop(0)]
    # Periodically try a random action, but not on the first trial in case the
    # user just wants a single reasonable recommendation.
    elif (
        self._llm_config.goal
        != sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE
        and len(self._history) > 1 and random.random() > 0.1):
      logging.info('##########################\n##### BAYESIAN OPT'
                   ' ######\n##########################')
      # selected_actions = self._random_action()
      dp = self._bayesian_opt.decision_point(request)
      selected_actions = {}
      for a in dp.action:
        selected_actions[a.key] = a.value.double_value
      selected_actions = [selected_actions]
      print('selected_actions=%s' % selected_actions)

    else:
      selected_actions = self._get_action(request.worker_id)

    logging.info('decision_point(): selected_actions=%s', selected_actions)

    self._history[-1]['action'] = selected_actions[0]
    # If there are more actions to perform, store them in self._actions_to_do
    if len(selected_actions) >= 1:
      self._actions_to_do.extend(selected_actions[1:])

    # self.script += '    {' + ', '.join([
    #     f'"{key}": {value}'
    #       for key, value in selected_actions.items()
    #     ]) + '}\n'

    for key, value in self._history[-1]['action'].items():
      # a = dp_response.action.add()
      # a.key = key
      # a.value.double_value = float(value)
      dp_response.action.params[key].CopyFrom(sight_pb2.Value(double_value=float(value),
                                                              sub_type = sight_pb2.Value.ST_DOUBLE))

    self._num_decision_points += 1

    self._lock.release()
    dp_response.action_type = (
        service_pb2.DecisionPointResponse.ActionType.AT_ACT)
    return dp_response
  

  @overrides
  def WorkerAlive(
      self, request: service_pb2.WorkerAliveRequest
  ) -> service_pb2.WorkerAliveResponse:
    method_name = "WorkerAlive"
    logging.debug(">>>>  In %s of %s", method_name, __file__)
    response = service_pb2.WorkerAliveResponse()
    response.status_type = service_pb2.WorkerAliveResponse.StatusType.ST_ACT
    decision_message = response.decision_messages.add()
    decision_message.action_id = 1
    logging.info("worker_alive_status is %s", response.status_type)
    logging.debug("<<<<  Out %s of %s", method_name, __file__)
    return response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    self._lock.acquire()

    logging.info('FinalizeEpisode request=%s', request)
    for i in range(len(request.decision_messages)):
      if len(request.decision_messages[i].decision_outcome.outcome_params.params) > 0:
        self._history[-1]['outcome'] = convert_proto_to_dict(
            request.decision_messages[i].decision_outcome.outcome_params)
      else:
        self._history[-1]['outcome'] = request.decision_messages[i].decision_outcome.reward
      # self.last_outcome = self._history[-1]['outcome']

      logging.info('self._history[-1]=%s', self._history[-1])
      request.decision_messages[i].decision_point.choice_params.CopyFrom(
          convert_dict_to_proto(dict=self._history[-1]['action']))
      self._bayesian_opt.finalize_episode(request)

      if (self._llm_config.goal ==
          sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE):
        # If there are no outstanding acitions, ask the LLM whether the user's
        # question can be answered via the already-completed model runs.
        if len(self._actions_to_do) == 0:
          can_respond_to_question, response = self._is_done(request.worker_id)
          self._response_ready = can_respond_to_question
          if self._response_ready:
            self._waiting_on_tell = True
            self._response_for_listen = response
    self._lock.release()

    logging.info(
        'FinalizeEpisode response=%s',
        service_pb2.FinalizeEpisodeResponse(response_str='Success!'),
    )
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def tell(self, request: service_pb2.TellRequest) -> service_pb2.TellResponse:
    tell_response = service_pb2.TellResponse()
    self._lock.acquire()
    logging.info('tell() request=%s', request)

    if (self._llm_config.goal
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE
        and self._waiting_on_tell):
      logging.info('INTERACTIVE')
      self._intro += '\n' + self._history_to_text(False) + '\n'
      self._history = []
      self._intro += 'User input: ' + request.message_str
      # self._intro += '\n' + request.message_str
      self._waiting_on_tell = False
      logging.info('tell self._intro=%s', self._intro)

    self._lock.release()
    tell_response.response_str = self._ask(self._intro)
    return tell_response

  @overrides
  def listen(self,
             request: service_pb2.ListenRequest) -> service_pb2.ListenResponse:
    listen_response = service_pb2.ListenResponse()
    self._lock.acquire()
    logging.info(
        'listen() request=%s, self._response_ready=%s,'
        ' self._response_for_listen=%s',
        request,
        self._response_ready,
        self._response_for_listen,
    )

    if (self._llm_config.goal ==
        sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE):
      listen_response.response_ready = self._response_ready
      if self._response_ready:
        listen_response.response_str = self._response_for_listen
      self._response_ready = False

    self._lock.release()
    logging.info('listen() response=%s', listen_response)
    return listen_response

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    bayesian_opt_status = self._bayesian_opt.current_status(request)
    return service_pb2.CurrentStatusResponse(
        response_str=f"""[LLM: script={self._intro + self._history_to_text(None)}
-----------------
BayesianOpt={bayesian_opt_status.response_str}""")
