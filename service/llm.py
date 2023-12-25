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
import random
import requests
import google.auth
import google.auth.transport.requests
import json

class LLM(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.

  Attributes:
    script: The script of the conversation accrued so far.
  """

  def __init__(self):
    super().__init__()
    genai.configure(api_key='AIzaSyCTXpaCJfIlWY4QO-QgpPf15dIbVFWKysI')
    self.model = genai.GenerativeModel('gemini-pro')
    self.script = ''
    self.num_dps = 0
    self.last_outcome = 0

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    response = super(LLM, self).launch(request)
    llm_config = request.decision_config_params.choice_config[request.label].llm_config

    self.script = 'You are controlling an agent that is trying to reach a goal. The agent is described as follows.\n'
    self.script += f'"{llm_config.description}"'
    self.script += 'The simulation will periodically report its state and then ask you ' + \
                  'to select an action for it to perform. After it has performed this ' + \
                  'action it will report back the numeric outcome of the this action. ' + \
                  'Higher outcome values are better than low outcome values beause higher outcomes mean that the temperature of the shower is closer to the ideal comfortable temperature. Your job ' + \
                  'is to choose actions that maximize the outcome values.\n' + \
                  'The state of the simulation consists of the following attributes: \n'
    self.script += '    {' + ', '.join([
          f'"{key}": {{ "description": {p.description}, "min_value": {p.min_value}, "max_value": {p.max_value} }},\n'
          for key, p in self.state.items()
        ]) + '}\n'
    self.script += 'The possible actions you need to select are: \n'
    self.script += '    {' + ', '.join([
          f'"{key}": {{ "description": {p.description}, "min_value": {p.min_value}, "max_value": {p.max_value} }},\n'
          for key, p in self.actions.items()
        ]) + '}\n'
    self.script += '========================\n'
    for i in range(5):
      self.script += 'Decision State:\n'
      self.script += '    {' + ', '.join([
          f'"{key}": {((p.max_value - p.min_value) * random.random() + p.min_value)}\n'
          for key, p in self.state.items()
        ]) + '}\n'
      self.script += 'Decision Action (json format):\n'
      self.script += '    {' + ', '.join([
          f'"{key}": {((p.max_value - p.min_value) * random.random() + p.min_value)}\n'
          for key, p in self.actions.items()
        ]) + '}\n'
      self.script += f'Decision Outcome: {random.random()}\n'
      self.script += '========================\n'
    
    response.display_string = 'LLM SUCCESS! '+self.script
    return response

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    logging.info('request=%s', request)
    addition = ''
    if self.num_dps > 0:
      addition += 'Decision Outcome: '+str(request.decision_outcome.outcome_value)+'\n'
      if self.last_outcome < request.decision_outcome.outcome_value-.1:
        addition += '  This is a better outcome than the last time.\n'
      elif self.last_outcome > request.decision_outcome.outcome_value+.1:
        addition += '  This is a worse outcome than the last time.\n'
      else:
        addition += '  This is a similar outcome to the last time.\n'
      addition += '========================\n'

    addition += 'Decision State:\n'
    addition += '    {' + ', '.join([
        f'"{p.key}": {p.value.double_value}\n'
          for p in request.decision_point.state_params
        ]) + '}\n'
    addition += 'Decision Action (json format):\n'

    logging.info('addition: %s', addition)
    self.script += addition

    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    r = requests.post(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/chorus-scout/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent", 
        data=
        json.dumps({
        "contents": {
            "role": "user",
            "parts": {
                "text": self.script
            },
        },
        "safety_settings": {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        },
        "generation_config": {
            "temperature": 0.9,
            "topP": 0.8,
            "topK": 3,
            "maxOutputTokens": 8192,
            # "stopSequences": [".", "?", "!"]
        }
        }),    
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json; charset=utf-8",
        })
    # logging.info('r=%s', r)
    response = r.json()
    logging.info('script: %s', self.script)
    logging.info('response=%s', response)
    # logging.info('text=%s', response[0]['candidates'][0]['content']['parts'][0]['text'])
    text = response[0]['candidates'][0]['content']['parts'][0]['text']
    if text[-1] != '}':
      text += '}'
    logging.info('text=%s', text)
    selected_actions = json.loads(text)

    self.script += '    {' + ', '.join([
        f'"{key}": {value}\n'
          for key, value in selected_actions.items()
        ]) + '}\n'
    
    dp_response = service_pb2.DecisionPointResponse()
    for key, value in selected_actions.items():
      a = dp_response.action.add()
      a.key = key
      a.value.double_value = float(value)

    # response = self.model.generate_content(self.script)
    # logging.info('genai response='+response)

    logging.info('dp_response=%s', dp_response)
    self.num_dps += 1
    self.last_outcome = request.decision_outcome.outcome_value
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    logging.info('Running for LLM....')

    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    return service_pb2.CurrentStatusResponse(response_str=f'[LLM: script={self.script}\n')
