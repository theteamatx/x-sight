from typing import Dict
from typing import Any
from sight.proto import sight_pb2

#? not used anymore
# def config_to_attr(config: Dict, key: str) -> Dict:
#   """convert dict's values into sight_pb2 object

#   Args:
#       config (Dict): value containing min,max
#       key (str): attribute type

#   Returns:
#       Dict: attribute dict with values as sight_pb2's object
#   """
#   attr_dict = {}
#   if key in config:
#     for key, params in config[key].items():
#       if params:
#         attr_dict[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
#           min_value=params['min_value'], max_value=params['max_value'])
#       else:
#         attr_dict[key] = sight_pb2.DecisionConfigurationStart.AttrProps()
#   return attr_dict


def attr_dict_to_proto(
    attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps],
    attrs_proto: Any,
):
  """Converts a dict of attribute constraints to its proto representation."""
  for attr_name, attr_details in attrs.items():
    attrs_proto[attr_name].CopyFrom(attr_details)

def description_to_proto(
    choice_config: Dict[str, sight_pb2.DecisionConfigurationStart.ChoiceConfig],
    choice_config_proto: Any,
):
  """Converts a dict of choice constraints to its proto representation."""
  for label, choice_details in choice_config.items():
    choice_config_proto[label].CopyFrom(choice_details)
