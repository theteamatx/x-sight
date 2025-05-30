from sight.worker.worker_helper import create_attr_props, create_choice_config


def get_question_label():
  return "Division"


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
  func_description = """This tool can perform Division operation on two integers using the Sight backend system.
  This function proposes action dictionary to the server via a Sight worker.
  It waits for the worker to process the action and return the computed result."""

  return create_choice_config(get_question_label(), func_description)
