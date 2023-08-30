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
import numpy as np

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
        return self.behavior_policy(self.Q[observation])    

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