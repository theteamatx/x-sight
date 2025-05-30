"""This is helper function to generate the textprotos for the models."""

import os
from typing import Any, Dict

from google.protobuf import text_format
from sight.proto import sight_pb2
from sight.widgets.decision import decision_helper
from worker.attribute_configs import calculator_attribute_config
from worker.attribute_configs import addition_attribute_config
from worker.attribute_configs import subtraction_attribute_config
from worker.attribute_configs import multiplication_attribute_config
from worker.attribute_configs import division_attribute_config



MessageToString = text_format.MessageToString
CONFIG_DIR = '.text_proto_configs'


def generate_the_text_protos(models: Dict[str, Any], output_dir_path: str):
  """Generate the textprotos for the models.

  Args:
    models: A dictionary of model names to their configuration.
    output_dir_path: The directory path to save the textprotos.
  """
  # create the textproto for the models we have
  os.makedirs(os.path.join(output_dir_path, CONFIG_DIR), exist_ok=True)
  for key, config in models.items():
    protodata = sight_pb2.DecisionConfigurationStart()
    # *** Config should have the this functions
    _ = hasattr(
        config, 'get_tool_description'
    ) and decision_helper.description_to_proto(
        config.get_tool_description(), protodata.choice_config
    )
    _ = hasattr(
        config, 'get_action_attrs'
    ) and decision_helper.attr_dict_to_proto(
        config.get_action_attrs(), protodata.action_attrs
    )
    _ = hasattr(
        config, 'get_outcome_attrs'
    ) and decision_helper.attr_dict_to_proto(
        config.get_outcome_attrs(), protodata.outcome_attrs
    )
    _ = hasattr(
        config, 'get_state_atrrs'
    ) and decision_helper.attr_dict_to_proto(
        config.get_state_atrrs(), protodata.state_attrs
    )

    file_name = os.path.join(output_dir_path, CONFIG_DIR, f'{key}.textproto')
    with open(file_name, 'w') as f:
      f.write(MessageToString(protodata))


if __name__ == '__main__':

  model_configs = {'addition': addition_attribute_config,
                   'subtraction': subtraction_attribute_config,
                   'multiplication': multiplication_attribute_config,
                   'division': division_attribute_config,
                   }
  current_dir = os.path.dirname(os.path.realpath(__file__))
  output_dir_path = os.path.join(current_dir, '..', '..', 'worker')
  generate_the_text_protos(model_configs, output_dir_path)

