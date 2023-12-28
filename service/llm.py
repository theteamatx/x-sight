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
import logging
from overrides import overrides
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from service import service_pb2
from service.optimizer_instance import param_dict_to_proto
from service.optimizer_instance import OptimizerInstance
from sight.proto import sight_pb2
import random
import requests
import google.auth
import google.auth.transport.requests
import json
import os

class LLM(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.

  Attributes:
    script: The script of the conversation accrued so far.
  """

  def __init__(self):
    super().__init__()
    genai.configure(api_key='AIzaSyCTXpaCJfIlWY4QO-QgpPf15dIbVFWKysI')
    self.model = genai.GenerativeModel('gemini-pro')
    self.intro = ''
    self.history = []
    self._history_len_for_prompt = 20
    self.num_dps = 0
    self.last_outcome = 0

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    response = super(LLM, self).launch(request)
    llm_config = request.decision_config_params.choice_config[request.label].llm_config

    self.intro = 'You are controlling an agent that is trying to reach a goal. The agent is described as follows.\n'
    self.intro += f'"{llm_config.description}"\n'
    self.intro += 'The simulation will periodically report its state and then ask you ' + \
                  'to select an action for it to perform. After it has performed this ' + \
                  'action it will report back the numeric outcome of the this action. ' + \
                  'Higher outcome values are better than low outcome values beause higher outcomes mean that the temperature of the shower is closer to the ideal comfortable temperature. Your job ' + \
                  'is to choose actions that maximize the outcome values.\n' + \
                  'The state of the simulation consists of the following attributes: \n'
    self.intro += '    {' + ', '.join([
          f'"{key}": {{ "description": {p.description}, "min_value": {p.min_value}, "max_value": {p.max_value} }},\n'
          for key, p in self.state.items()
        ]) + '}\n'
    self.intro += 'The possible actions you need to select are: \n'
    self.intro += '    {' + ', '.join([
          f'"{key}": {{ "description": {p.description}, "min_value": {p.min_value}, "max_value": {p.max_value} }},\n'
          for key, p in self.actions.items()
        ]) + '}\n'
    self.intro += '========================\n'
    
    response.display_string = 'LLM SUCCESS! '+self.intro
    logging.info('self.intro=%s', self.intro)
    return response

  def _random_action(self) -> Dict[str, float]:
    """Returns a random action."""
    a = {}
    for key, p in self.actions.items():
      a[key] = (p.max_value - p.min_value) * random.random() + p.min_value
    return a

  def _filtered_history(self) -> List[Any]:
    ordered_history = self.history[0:-1]
    # logging.info('#hist=%d ordered_history[#%d]=%s', len(self.history), len(ordered_history), ordered_history)
    ordered_history = sorted(ordered_history, key=lambda h: -h['outcome'])# if h['outcome'] is not None else 1e100)
    if len(ordered_history) > self._history_len_for_prompt:
      ordered_history = ordered_history[0:self._history_len_for_prompt-1]
    random.shuffle(ordered_history)
    logging.info('ordered_history[#%d]=%s', len(ordered_history), [h['outcome'] for h in ordered_history])
    return ordered_history+[self.history[-1]]


  def _history_to_text(self) -> str:
    t = ''
    self.last_outcome = None
    for h in self._filtered_history():
      t += 'Decision State:\n'
      t += '    {' + ', '.join([
          f'"{k}": {v}'
            for k, v in h['state'].items()
          ]) + '}\n'
      t += 'Decision Action (json format):\n'
      if h['action'] is not None:
        t += '    {' + ', '.join([
          f'"{key}": {value}'
            for key, value in h['action'].items()
          ]) + '}\n'
      if h['outcome'] is not None:
        t += 'Decision Outcome: '+str(h['outcome'])+'\n'
        if self.last_outcome is not None:
          if self.last_outcome < h['outcome']-.1:
            t += '  This is a better outcome than the last time.\n'
          elif self.last_outcome > h['outcome']+.1:
            t += '  This is a worse outcome than the last time.\n'
          else:
            t += '  This is a similar outcome to the last time.\n'
        t += '========================\n'
        self.last_outcome = h['outcome']
    return t

  def _history_to_chat(self) -> List[Dict[str, str]]:
    chat = []
    last_outcome = None
    last_outcome_message = ''
    for h in self._filtered_history():
      chat.append(
        {
          'author': 'USER',
          'content': last_outcome_message + 'Decision State:\n'+'    {' + ', '.join([
            f'"{k}": {v}'
              for k, v in h['state'].items()
            ]) + '}\n'+'Please provide the Decision Action (json format):\n',
        }
      )
      if h['action'] is not None:
        chat.append(
          {
            'author': 'AI',
            'content': '{' + ', '.join([
            f'"{key}": {value}'
              for key, value in h['action'].items()
            ]) + '}'
          }
        )
      if h['outcome'] is not None:
        last_outcome_message = 'Decision Outcome: '+str(h['outcome'])+'\n'
        if self.last_outcome is not None:
          if self.last_outcome < h['outcome']-.1:
            last_outcome_message += '  This is a better outcome than the last time.\n'
          elif self.last_outcome > h['outcome']+.1:
            last_outcome_message += '  This is a worse outcome than the last time.\n'
          else:
            last_outcome_message += '  This is a similar outcome to the last time.\n'
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
  
  def _ask_palm2_text(self) -> Dict[str, float]:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/text-bison-32k:predict",
          data = json.dumps(
            {
              "instances": [
                {
                  "prompt": self.intro + self._history_to_text()
                }
              ],
              "parameters": {
                "temperature": .2,
                "maxOutputTokens": 2048,
                "topK": 40,
                "topP": .55,
                # "groundingConfig": string,
                # "stopSequences": [ string ],
                # "candidateCount": integer,
                # "logprobs": integer,
                # "presencePenalty": float,
                # "frequencyPenalty": float,
                # "logitBias": map<string, float>,
                "echo": False
              }
            }),
            headers=self._get_req_headers()).json()
      logging.info('response=%s', response)
      if 'error' in response or response['predictions'][0]['content'].strip() == '':
        continue
      text = response['predictions'][0]['content'].split('\n')[0].strip()
      logging.info('history: %s', self._history_to_text())
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  def _ask_palm2_chat(self) -> Dict[str, float]:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/chat-bison-32k:predict",
          data = json.dumps(
            {
              "instances": [
                {
                  "context":  self.intro,
                  "messages": self.__history_to_chat(),
                },
              ],
              "parameters": {
                "temperature": .2,
                "maxOutputTokens": 2048,
                "topK": 40,
                "topP": .55,
                # "groundingConfig": string,
                # "stopSequences": [ string ],
                # "candidateCount": integer,
                # "logprobs": integer,
                # "presencePenalty": float,
                # "frequencyPenalty": float,
                # "logitBias": map<string, float>,
                "echo": False
              }
            }),
            headers=self._get_req_headers()).json()
      logging.info('response=%s', response)
      if 'error' in response or response['predictions'][0]['content'].strip() == '':
        continue
      text = response['predictions'][0]['content'].split('\n')[0].strip()
      logging.info('history: %s', self._history_to_text())
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  def _ask_geminipro(self) -> Dict[str, float]:
    while True:
      response = requests.post(
          f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ['PROJECT_ID']}/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent", 
          data= json.dumps(
            {
          "contents": {
              "role": "user",
              "parts": {
                  "text": self.intro + self._history_to_text()
              },
          },
          "safety_settings": {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "threshold": "BLOCK_LOW_AND_ABOVE"
          },
          "generation_config": {
              "temperature": 0.3,
              "topP": 0.8,
              "topK": 3,
              "maxOutputTokens": 8192,
              # "stopSequences": [".", "?", "!"]
          }
          }),    
          headers=self._get_req_headers()).json()
      logging.info('response=%s', response)
      # if 'error' in response or response['predictions'][0]['content'].strip() == '':
      if len(response) == 0:
        continue
      logging.info('history: %s', self._history_to_text())
      logging.info('text=%s', response[0]['candidates'][0]['content']['parts'][0]['text'])
      text = ''
      for r in response:
        text += r['candidates'][0]['content']['parts'][0]['text']
      if text == '':
        continue
      if text[-1] != '}':
        text += '}'
      logging.info('text=[%s]', text)
      try:
        return json.loads(text)
      except json.decoder.JSONDecodeError:
        continue

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    # logging.info('DecisionPoint request=%s', request)
    # self._append_outcome(request.decision_outcome.outcome_value)
    if len(self.history)>0:
      self.history[-1]['outcome'] = request.decision_outcome.outcome_value

    # self.script += 'Decision State:\n'
    # self.script += '    {' + ', '.join([
    #     f'"{p.key}": {p.value.double_value}'
    #       for p in request.decision_point.state_params
    #     ]) + '}\n'
    # self.script += 'Decision Action (json format):\n'
    self.history.append({
      'state': self._params_to_dict(request.decision_point.state_params),
      'action': None,
      'outcome': None,
    })

    if random.random() > .5:
      logging.info('##########################\n##### RANDOM ######\n##########################')
      selected_actions = self._random_action()
    else:
      selected_actions = self._ask_palm2_text()
      
    self.history[-1]['action'] = selected_actions

    # self.script += '    {' + ', '.join([
    #     f'"{key}": {value}'
    #       for key, value in selected_actions.items()
    #     ]) + '}\n'
    
    dp_response = service_pb2.DecisionPointResponse()
    for key, value in selected_actions.items():
      a = dp_response.action.add()
      a.key = key
      a.value.double_value = float(value)

    # response = self.model.generate_content(self.script)
    # logging.info('genai response='+response)

    # logging.info('dp_response=%s', dp_response)
    self.num_dps += 1
    self.last_outcome = request.decision_outcome.outcome_value
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    # logging.info('FinalizeEpisode request=', request)
    # self._append_outcome(request.decision_outcome.outcome_value)
    self.history[-1]['outcome'] = request.decision_outcome.outcome_value
    self.last_outcome = request.decision_outcome.outcome_value
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    return service_pb2.CurrentStatusResponse(response_str=f'[LLM: script={self.intro + self._history_to_text()}\n')
