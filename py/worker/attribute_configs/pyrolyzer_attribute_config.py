from sight.worker.worker_helper import create_attr_props, create_choice_config


def get_question_label():
  return "Calculator"


def get_action_attrs():
  """Returns the action attributes for the FVS action.
  """
  action_config = {
      'X': {
          'description': 'The first variable.',
          'type': 'integer',
          'min_value': 1,
          'max_value': 5
      },
      'Y': {
          'description': 'The second variable.',
          'type': 'integer',
          'min_value': 1,
          'max_value': 5
      },
      'Z': {
          'description': 'The third variable.',
          'type': 'integer',
          'min_value': 1,
          'max_value': 5
      }
  }
  return create_attr_props(action_config)


def get_outcome_attrs():
  """Returns the outcome attributes for the FVS outcome.
  """
  outcome_config = {'final_result': None}
  return create_attr_props(outcome_config)
