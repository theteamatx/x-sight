"""decision decorator to be used in worker function."""
from helpers.logs.logs_handler import logger as logging
from sight.widgets.decision import decision

# not used anymore, replaced this with run_generic method
def decision_worker(question_label):
  """Decorator that adds a decision API to the Sight log.

  This decorator is used to add a decision point to the Sight log before
  executing a function. It takes a question label as an argument, which
  is used to identify the decision point.

  The decorated function should:
    - take a Sight object as the first argument.
    - take the action selected by the user as the second argument.
    - return a tuple containing the reward and the outcome of the decision.

  Example:
    @decision_worker('question_label')
    def my_function(sight, action):
      # ... do something with the action ...
      reward = 1 if action == 'option A' else 0
      outcome = 'Success' if reward == 1 else 'Failed'
      return reward, outcome

    # Usage:
    my_outcome = my_function(sight_object)

  Args:
    question_label: The question that is asked to the user.

  Returns:
    A decorator that adds a decision point to the Sight log.
  """

  def decorator(func):
    def wrapper(sight):
      """Wrapper that adds a decision point to the Sight log.

      Args:
        sight: The Sight object.
      """
      action = decision.decision_point(question_label, sight)
      logging.info('actions we got : %s', action)
      reward, outcome = func(sight, action)
      logging.info('reward we got : %s', reward)
      logging.info('outcome we got : %s', outcome)
      decision.decision_outcome('decisionin_outcome', sight, reward, outcome)
    return wrapper
  return decorator
