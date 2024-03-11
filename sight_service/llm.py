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
import logging
import os
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import google.auth.transport.requests
import google.generativeai as genai
from overrides import overrides
import requests
from sight.proto import sight_pb2
from sight_service.bayesian_opt import BayesianOpt
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.proto import service_pb2

_GENAI_API_KEY = os.environ['GENAI_API_KEY']


class LLM(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.

  Attributes:
    script: The script of the conversation accrued so far.
  """

  def __init__(self):
    super().__init__()
    genai.configure(api_key=_GENAI_API_KEY)
    self.intro = ''
    self.history = {}
    self._history_len_for_prompt = 20
    self.num_dps = 0
    self.last_outcome = 0
    self._lock = threading.RLock()

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    response = super(LLM, self).launch(request)
    self._llm_config = request.decision_config_params.choice_config[
        request.label
    ].llm_config
    self._bayesian_opt = BayesianOpt()
    self._bayesian_opt.launch(request)

    self.intro += ''
    if (
        self._llm_config.goal
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_OPTIMIZE
    ):
      self.intro = (
          'You are controlling an agent that is trying to reach a goal. The'
          ' agent is described as follows.\n'
      )
    self.intro += f'"{self._llm_config.description}"\n'
    if (
        self._llm_config.goal
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_OPTIMIZE
    ):
      self.intro += (
          'The simulation will periodically report its state and then ask you '
          + 'to select an action for it to perform. After it has performed'
          ' this '
          + 'action it will report back the numeric outcome of the this'
          ' action. '
          + 'Higher outcome values are better than low outcome values. Your'
          ' job '
          + 'is to choose actions that maximize the outcome values.\n'
      )
    if len(self.state) > 0:
      self.intro += (
          'The state of the simulation consists of the following attributes: \n'
      )
      self.intro += (
          '  {\n    '
          + '\n    '.join([
              f'"{key}": {{ "description": {p.description}, "min_value":'
              f' {p.min_value}, "max_value": {p.max_value} }},'
              for key, p in self.state.items()
          ])
          + '}\n'
      )
    self.intro += 'The possible actions you need to select are: \n'
    self.intro += (
        '  {\n    '
        + '\n    '.join([
            f'"{key}": {{ "description": {p.description}, "min_value":'
            f' {p.min_value}, "max_value": {p.max_value} }},'
            for key, p in self.actions.items()
        ])
        + '}\n'
    )
    self.intro += '========================\n'

    detail_prompt = (
        'Please summarize everything you know about these parameters for the'
        ' above application area, detail the steps that need to be taken to'
        ' create a good estimate these parameters.\n'
    )
    if (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_TEXT_BISON
    ):
      text = self._ask_text_bison(self.intro + detail_prompt)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_CHAT_BISON
    ):
      text = self._ask_chat_bison(self.intro + detail_prompt)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_GEMINI_PRO
    ):
      text = self._ask_gemini_pro(self.intro + detail_prompt)
    self.intro += detail_prompt + text + '\n'

    detail_prompt = (
        'Based on this plan describe the most reasonable estimate of these'
        ' parameters\n'
    )
    if (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_TEXT_BISON
    ):
      text = self._ask_text_bison(self.intro + detail_prompt)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_CHAT_BISON
    ):
      text = self._ask_chat_bison(self.intro + detail_prompt)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_GEMINI_PRO
    ):
      text = self._ask_gemini_pro(self.intro + detail_prompt)
    self.intro += detail_prompt + text + '\n'

    response.display_string = 'LLM SUCCESS! ' + self.intro
    logging.info('self.intro=%s', self.intro)
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

  def _random_event(self) -> Dict[str, Any]:
    return {
        'state': self._random_state(),
        'action': self._random_action(),
        'outcome': random.random(),
    }

  def _filtered_history(self, worker_id: Optional[str]) -> List[Any]:
    ordered_history = []
    for worker_hist in self.history.values():
      ordered_history.extend(worker_hist[0:-1])
    logging.info('#hist=%d ordered_history[#%d]=%s', len(self.history), len(ordered_history), ordered_history)
    ordered_history = sorted(
        ordered_history, key=lambda h: -h['outcome']
    )  # if h['outcome'] is not None else 1e100)
    if len(ordered_history) > self._history_len_for_prompt:
      ordered_history = ordered_history[0 : self._history_len_for_prompt - 1]
    random.shuffle(ordered_history)

    # If this is the first question, add a random event to serve as an example of the format.
    if len(ordered_history) == 0:
      ordered_history.append(self._random_event())

    logging.info(
        'ordered_history[#%d]=%s',
        len(ordered_history),
        [h['outcome'] for h in ordered_history],
    )
    if worker_id is None:
      return ordered_history
    return ordered_history + [self.history[worker_id][-1]]

  def _history_to_text(self, worker_id: Optional[str]) -> str:
    t = ''
    self.last_outcome = None
    for h in self._filtered_history(worker_id):
      t += 'Decision State:\n'
      t += (
          '    {'
          + ', '.join([f'"{k}": {v}' for k, v in h['state'].items()])
          + '}\n'
      )
      t += 'Decision Action (json format):\n'
      if h['action'] is not None:
        t += (
            '    {'
            + ', '.join(
                [f'"{key}": {value}' for key, value in h['action'].items()]
            )
            + '}\n'
        )
      if h['outcome'] is not None:
        t += 'Decision Outcome: ' + str(h['outcome']) + '\n'
        if self.last_outcome is not None:
          if self.last_outcome < h['outcome'] - 0.1:
            t += '  This is a better outcome than the last time.\n'
          elif self.last_outcome > h['outcome'] + 0.1:
            t += '  This is a worse outcome than the last time.\n'
          else:
            t += '  This is a similar outcome to the last time.\n'
        t += '========================\n'
        self.last_outcome = h['outcome']
    return t

  def _history_to_chat(self, worker_id: str) -> List[Dict[str, str]]:
    chat = []
    last_outcome = None
    last_outcome_message = ''
    for h in self._filtered_history(worker_id):
      chat.append({
          'author': 'USER',
          'content': (
              last_outcome_message
              + 'Decision State:\n'
              + '    {'
              + ', '.join([f'"{k}": {v}' for k, v in h['state'].items()])
              + '}\n'
              + 'Please provide the Decision Action (json format):\n'
          ),
      })
      if h['action'] is not None:
        chat.append({
            'author': 'AI',
            'content': (
                '{'
                + ', '.join(
                    [f'"{key}": {value}' for key, value in h['action'].items()]
                )
                + '}'
            ),
        })
      if h['outcome'] is not None:
        last_outcome_message = 'Decision Outcome: ' + str(h['outcome']) + '\n'
        if self.last_outcome is not None:
          if self.last_outcome < h['outcome'] - 0.1:
            last_outcome_message += (
                '  This is a better outcome than the last time.\n'
            )
          elif self.last_outcome > h['outcome'] + 0.1:
            last_outcome_message += (
                '  This is a worse outcome than the last time.\n'
            )
          else:
            last_outcome_message += (
                '  This is a similar outcome to the last time.\n'
            )
    return chat

  def _params_to_dict(self, dp: sight_pb2) -> Dict[str, float]:
    """Returns the dict representation of a DecisionParams proto"""
    d = {}
    for a in dp:
      d[a.key] = a.value.double_value
    return d

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

  def _ask_text_bison(self, prompt) -> str:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/text-bison-32k:predict",
          data=json.dumps({
              'instances': [{'prompt': prompt}],
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
      logging.info('response=%s', response)
      if (
          'error' in response
          or response['predictions'][0]['content'].strip() == ''
      ):
        continue
      return response['predictions'][0]['content'].strip()

  def _action_from_text_bison(self, worker_id: str) -> Dict[str, float]:
    logging.info('ask_text_bison')
    logging.info(self.intro + self._history_to_text(worker_id))
    while True:
      text = self._ask_text_bison(self.intro + self._history_to_text(worker_id))
      logging.info('text=[%s]', text)
      # text = text.removeprefix('```json\n')
      # logging.info('text=[%s]', text)
      text = text.strip('`').split('\n')[0]
      # text = text.split('\n')[0].strip()
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  def _ask_chat_bison(self, prompt, message) -> str:
    response = requests.post(
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/chat-bison-32k:predict",
        data=json.dumps({
            'instances': [
                {
                    'context': prompt,
                    'messages': message,
                },
            ],
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
    logging.info('response=%s', response)
    logging.info(
        "response['predictions']=%s", response['predictions'][0]['candidates']
    )
    # if 'error' in response or response['predictions'][0]['content'].strip() == '':
    #  continue
    return response['predictions'][0]['candidates'][0]['content'].strip()

  def _action_from_chat_bison(self, worker_id: str) -> Dict[str, float]:
    while True:
      text = self._ask_chat_bison(self.intro, self._history_to_chat(worker_id))
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  def _ask_gemini_pro(self, prompt) -> str:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent",
          data=json.dumps({
              'contents': {
                  'role': 'user',
                  'parts': {'text': prompt},
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
      logging.info('response=%s', response)
      if len(response) == 0:
        continue
      text = ''
      for r in response:
        if 'parts' in r['candidates'][0]['content']:
          text += r['candidates'][0]['content']['parts'][0]['text']
      text = text.strip()
      if text == '':
        continue
      return text

  def _action_from_gemini_pro(self, worker_id: str) -> Dict[str, float]:
    while True:
      logging.info('ask_geminipro')
      logging.info(self.intro + self._history_to_text(worker_id))
      text = self._ask_gemini_pro(self.intro + self._history_to_text(worker_id))
      logging.info('text=[%s]', text.split('\n'))
      text = text.split('\n')[0]
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    logging.info('DecisionPoint request=%s', request)
    # self._append_outcome(request.decision_outcome.outcome_value)
    self._lock.acquire()

    if request.worker_id not in self.history:
      self.history[request.worker_id] = []
    if len(self.history[request.worker_id]) > 0 and 'outcome' not in self.history[request.worker_id][0]:
      self.history[request.worker_id][-1]['outcome'] = request.decision_outcome.outcome_value

    # self.script += 'Decision State:\n'
    # self.script += '    {' + ', '.join([
    #     f'"{p.key}": {p.value.double_value}'
    #       for p in request.decision_point.state_params
    #     ]) + '}\n'
    # self.script += 'Decision Action (json format):\n'
    self.history[request.worker_id].append({
        'state': self._params_to_dict(request.decision_point.state_params),
        'action': None,
        'outcome': None,
    })

    print('ALGORITHM=%s' % self._llm_config.algorithm)
    # Periodically try a random action, but not on the first trial in case the 
    # user just wants a single reasonable recommendation.
    if len(self.history[request.worker_id]) > 1 and random.random() > 0.1:
      logging.info(
          '##########################\n##### RANDOM'
          ' ######\n##########################'
      )
      # selected_actions = self._random_action()
      dp = self._bayesian_opt.decision_point(request)
      selected_actions = {}
      for a in dp.action:
        selected_actions[a.key] = a.value.double_value
      print ('selected_actions=%s' % selected_actions)

    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_TEXT_BISON
    ):
      selected_actions = self._action_from_text_bison(request.worker_id)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_CHAT_BISON
    ):
      selected_actions = self._action_from_chat_bison(request.worker_id)
    elif (
        self._llm_config.algorithm
        == sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_GEMINI_PRO
    ):
      selected_actions = self._action_from_gemini_pro(request.worker_id)

    self.history[request.worker_id][-1]['action'] = selected_actions

    # self.script += '    {' + ', '.join([
    #     f'"{key}": {value}'
    #       for key, value in selected_actions.items()
    #     ]) + '}\n'

    dp_response = service_pb2.DecisionPointResponse()
    for key, value in selected_actions.items():
      a = dp_response.action.add()
      a.key = key
      a.value.double_value = float(value)

    self.num_dps += 1
    self.last_outcome = request.decision_outcome.outcome_value

    self._lock.release()
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    self._lock.acquire()

    logging.info('FinalizeEpisode request=%s', request)
    self.history[request.worker_id][-1]['outcome'] = \
        request.decision_outcome.outcome_value
    self.last_outcome = request.decision_outcome.outcome_value
    logging.info('FinalizeEpisode self._bayesian_opt=%s', self._bayesian_opt)
    logging.info('self.history[-1]=%s', self.history[request.worker_id][-1])
    for key, value in self.history[request.worker_id][-1]['action'].items():
      a = request.decision_point.choice_params.add()
      a.key = key
      a.value.double_value = float(value)
    self._bayesian_opt.finalize_episode(request)

    self._lock.release()

    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    bayesian_opt_status = self._bayesian_opt.current_status(request)
    return service_pb2.CurrentStatusResponse(
        response_str=f'''[LLM: script={self.intro + self._history_to_text(None)}
-----------------
BayesianOpt={bayesian_opt_status.response_str}'''
    )
