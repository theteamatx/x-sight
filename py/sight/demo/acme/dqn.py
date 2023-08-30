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

from acme.agents.tf import dqn
from acme import specs
import dm_env
import haiku as hk
import jax
import numpy as np
import sonnet as snt

# from py.demo.acme.grid_world import build_gridworld_task,ObservationType,setup_environment

# grid = build_gridworld_task(
#     task='simple', 
#     observation_type=ObservationType.GRID,
#     max_episode_length=200)

# environment, environment_spec = setup_environment(grid)

print('*'*30)
# environment_spec = specs.EnvironmentSpec(
#                         observations=specs.Array(shape=(1,), dtype=np.float32, name='observation'),
#                         actions=specs.DiscreteArray(10, dtype=np.int32, name='action'),
#                         rewards=specs.Array(shape=(), dtype=np.float32, name='reward'),
#                         discounts= specs.BoundedArray(shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')
#                     )
environment_spec = specs.EnvironmentSpec(
    observations=specs.BoundedArray(shape=(1,), dtype=np.float32, minimum=0, maximum=9, name='observation'),
    actions=specs.BoundedArray(shape=(), dtype=np.int32, minimum=0, maximum=9, name='action'),
    rewards=specs.Array(shape=(), dtype=np.float32, name='reward'),
    discounts= specs.BoundedArray(shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')
)
print("environment_spec : ", environment_spec)
print("environment_spec type: ", type(environment_spec))
print('*'*30)

# environment_spec = EnvironmentSpec(
#                     observations=Array(shape=(9, 10, 3), dtype=dtype('float32'), name='observation_grid'), 
#                     actions=DiscreteArray(shape=(), dtype=int32, name=action, minimum=0, maximum=3, num_values=4), 
#                     rewards=Array(shape=(), dtype=dtype('float32'), name='reward'), 
#                     discounts=BoundedArray(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0)
#                 )

# def network(x):
#   model = hk.Sequential([
#       hk.Conv2D(32, kernel_shape=[4,4], stride=[2,2], padding='VALID'),
#       jax.nn.relu,
#       hk.Conv2D(64, kernel_shape=[3,3], stride=[1,1], padding='VALID'),
#       jax.nn.relu,
#       hk.Flatten(),
#       hk.nets.MLP([50, 50, environment_spec.actions.num_values])
#   ])
#   return model(x)
# network = hk.without_apply_rng(hk.transform(network))

# network = snt.Sequential([])
network = snt.Sequential([
    snt.Linear(1),
    # tf.nn.relu,
    # snt.Linear(1),
])

# print("environment_spec : ", environment_spec)
# print("network : ", network)
print('*'*30)
agent  = dqn.DQN(environment_spec, network)


print("agent : ", agent)
# print(agent.__dict__)
print('*'*30)

timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,        #decision outcome (joy value)
        discount=None,
        observation=np.zeros(1, dtype=np.float32))
agent.observe_first(timestep)
print('*'*30)

# timestep = environment.reset()
# print("timestep : ", timestep)

# timestep = dm_env.TimeStep(
#         step_type=dm_env.StepType.FIRST,
#         reward=None,
#         discount=None,
#         observation=np.ones(1, dtype=np.float32)*100)
print("timestep : ", timestep)

selected_action = agent.select_action(timestep.observation)
print("selected_action : ", selected_action)
print("selected_action type: ", type(selected_action))
print('*'*30)

# # player_timestep = self._get_player_timestep(timestep, player)
# timestep = environment.step(selected_action)

timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=np.array(101., dtype=np.float32),        #decision outcome (joy value)
        discount=np.array(1., dtype=np.float32),
        observation=np.zeros(1, dtype=np.float32))

# print("timestep : ",timestep)
# print() 
agent.observe(selected_action,timestep)

print('*'*30)

agent.update()

print('*'*60)

# if self._should_update:
#     self._actors[player].update()

