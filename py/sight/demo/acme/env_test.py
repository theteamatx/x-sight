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

from py.demo.acme.acme_environment import SweetnessEnvAcme
from py.demo.acme.acme_agent import SweetnessAgentAcme

import acme
from acme.utils import loggers
# from acme import EnvironmentLoop
# from dm_env import StepType


# env_loop_logger = loggers.InMemoryLogger()
env_loop_logger = loggers.TerminalLogger()

loop = acme.EnvironmentLoop(SweetnessEnvAcme(), 
                        SweetnessAgentAcme(),
                        logger=env_loop_logger
                    )
# print("loop : ",loop)
# print("type(loop) : ",type(loop))
loop.run(num_episodes=10)



# obj = SweetnessEnvAcme()

# reset = obj.reset()
# print(reset)


# while(True):
#     print("action = ", action)
#     step = obj.step(action)
#     print("step = ", step)

#     if(step.step_type==StepType.LAST):
#         break
#     print('*'*30)
