
from typing import Dict
from sight.proto import sight_pb2

def config_to_attr(config : Dict, key : str) -> Dict:
  """convert dict's values into sight_pb2 object

  Args:
      config (Dict): value containing min,max
      key (str): attribute type

  Returns:
      Dict: attribute dict with values as sight_pb2's object
  """
  attr_dict = {}
  if key in config:
    for key, params in config[key].items():
      attr_dict[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
              min_value=params['min_value'],
              max_value=params['max_value']
          )
  return attr_dict
