from sight.worker.worker_helper import create_attr_props, create_choice_config

def get_question_label():
  return "Calculator"

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


def get_tool_description():
  """Returns tool description
  """
  func_description = """This tool can perform a basic arithmetic operation (addition, subtraction, multiplication and division) on two integers using the Sight backend system.
  This function proposes a calculation action (with action dictionary as input containing all action parameter as key value pairs) to the server via a Sight worker.
  It waits for the worker to process the action and return the computed result."""

  return create_choice_config(get_question_label(), func_description)

