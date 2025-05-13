import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn
import json
import os

from absl import app
from absl import flags
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight_service.proto import service_pb2

FLAGS = flags.FLAGS


# Define the black box function to optimize.
def black_box_function(args):
  return sum(xi**2 for xi in args)


def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """

  next_point = decision.decision_point("label", sight)

  reward = black_box_function(list(next_point.values()))
  outcome = {'sum': 30, 'avg': 10}

  decision.decision_outcome(json.dumps(next_point), sight, reward, outcome)


def get_sight_instance():
  print('creating sight object')
  params = sight_pb2.Params(
      label='original_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(driver_fn=driver,
                 state_attrs={
                     'state_1':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=100,
                         ),
                     'state_2':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=100,
                         ),
                     'state_3':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=100,
                         )
                 },
                 action_attrs={
                     'action_1':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=10,
                         ),
                     'action_2':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=10,
                         ),
                     'action_3':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=10,
                         )
                 },
                 sight=sight,
                 outcome_attrs={
                     'sum':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=10000,
                             description='The sum of choosen action params.',
                         ),
                     'avg':
                         sight_pb2.DecisionConfigurationStart.AttrProps(
                             min_value=0,
                             max_value=100,
                             description='The avg of choosen action params.',
                         )
                 })


if __name__ == "__main__":
  app.run(main)
