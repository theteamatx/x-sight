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

import acme
import gym
# from gym import wrappers
from acme import wrappers
import numpy as np
import random


class RandomAgent(acme.Actor):
    """A random agent for the Black Jack environment."""
    
    def __init__(self):
        
        # init action values, will not be updated by random agent
        self.Q = np.zeros((32,11,2,2))
        
        # specify the behavior policy
        self.behavior_policy = lambda q_values: np.random.choice(2)
        
        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None
        
    def select_action(self, observation):
        "Choose an action according to the behavior policy."
        return self.behavior_policy(self.Q[observation[0]])    

    def observe_first(self, timestep):
        "Observe the first timestep." 
        self.timestep = timestep

    def observe(self, action, next_timestep):
        "Observe the next timestep."
        self.action = action
        self.next_timestep = next_timestep
        
    def update(self, wait = False):
        "Update the policy."
        # no updates occur here, it's just a random policy
        self.timestep = self.next_timestep 



env = wrappers.GymWrapper(gym.make('Blackjack-v1'))
agent = RandomAgent()
global_action_list = []
global_reward_list = []
random.seed(0)


# repeat for a number of episodes
for episode in range(10):
    action_list = []
    reward_list = []

    # print('*'*10)
    # make first observation
    timestep = env.reset()
    agent.observe_first(timestep)
    # print("timestep : ", timestep)

    # run an episode
    while not timestep.last():
        # print("in inner while loop......... ")

        # generate an action from the agent's policy
        # action = agent.select_action(timestep.observation)\
        action = random.randint(0, 1)
        action_list.append(action)
        # print("action : ", action)
        
        # step the environment
        timestep = env.step(action)
        # print("timestep : ", timestep)

        reward_list.append(timestep.reward)
        # have the agent observe the next timestep
        agent.observe(action, next_timestep=timestep)
        
        # let the agent perform updates
        agent.update()
    
    global_action_list.append(action_list)
    global_reward_list.append(reward_list)

print("action_list : ", global_action_list)
print("reward_list : ", global_reward_list)