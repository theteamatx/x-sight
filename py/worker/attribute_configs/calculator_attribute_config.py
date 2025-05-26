from worker.helper import create_attr_props


def get_action_attrs():
  """Returns the action attributes for the FVS action.
  """
  action_config = {
      'operand1': {
          'description': 'The first integer operand.',
          'type': 'integer'
      },
      'operand2': {
          'description': 'The second integer operand.',
          'type': 'integer'
      },
      'operator': {
          'description':
              'The operation to perform. Must be one of: add, subtract, multiply, divide.',
          'type':
              'string',
      }
  }
  return create_attr_props(action_config)


def get_outcome_attrs():
  """Returns the outcome attributes for the FVS outcome.
  """
  outcome_config = {'final_result': None}
  return create_attr_props(outcome_config)
