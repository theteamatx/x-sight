from typing import Any
from sight.proto import sight_pb2
from absl import flags
from sight.widgets.decision import utils
import os
from pathlib import Path
from sight.widgets.decision.utils import get_config_dir_path
from google.protobuf import text_format

FLAGS = flags.FLAGS

DATA_TYPE_MAP = {
    "integer": sight_pb2.DecisionConfigurationStart.DT_INT64,
    "float": sight_pb2.DecisionConfigurationStart.DT_FLOAT32,
    "double": sight_pb2.DecisionConfigurationStart.DT_FLOAT64,
    "string": sight_pb2.DecisionConfigurationStart.DT_STRING,
}


def create_attr_props(
    config_dict: dict[str, Any],
) -> dict[str, sight_pb2.DecisionConfigurationStart.AttrProps]:
  """Create AttrProps for each key in the config_dict.

  Args:
    config_dict (dict[str, Any]): Config dict to be converted to AttrProps.

  Returns:
    dict[str, sight_pb2.DecisionConfigurationStart.AttrProps]:
    AttrProps for each key in the config_dict.
  """
  # return {
  #     key: (sight_pb2.DecisionConfigurationStart.AttrProps() if value
  #           is not None else sight_pb2.DecisionConfigurationStart.AttrProps()
  #          ) for key, value in config_dict.items()
  # }

  attr_prop_dict = {}
  for key, value in config_dict.items():
    value_proto = sight_pb2.DecisionConfigurationStart.AttrProps()
    # print('value : ', value, type(value))

    if value is not None:
      if 'description' in value:
        value_proto.description = value['description']
      if 'type' in value:
        data_type_str = value['type'].lower()
        if data_type_str in DATA_TYPE_MAP:
          value_proto.data_type = DATA_TYPE_MAP[data_type_str]
        else:
          raise ValueError(f"Unknown data type: {data_type_str}")

    attr_prop_dict[key] = value_proto

  return attr_prop_dict


def get_text_proto_data(question_label):
  questions_info = utils.load_yaml_config(get_config_dir_path() +
                                          '/question_config.yaml')

  if (question_label not in questions_info):
    raise ValueError(f"Unknown question label: {question_label}")

  relative_text_proto_path = questions_info[question_label]['attrs_text_proto']
  if os.path.exists(relative_text_proto_path):
    with open(relative_text_proto_path, 'r') as f:
      text_proto_data = f.read()
  else:
    current_file = Path(__file__).resolve()
    sight_repo_path = current_file.parents[4]

    absolute_text_proto_path = sight_repo_path.joinpath(
        relative_text_proto_path)
    print('absolute_text_proto_path : ', absolute_text_proto_path)
    print('relative_text_proto_path : ', relative_text_proto_path)

    if not os.path.exists(absolute_text_proto_path):
      raise FileNotFoundError(f'File not found {relative_text_proto_path}')

    with open(absolute_text_proto_path, 'r') as f:
      text_proto_data = f.read()

  return text_proto_data


def get_description_from_textproto(question_label) -> tuple[str, str]:
  # we get text_proto data in string type
  text_proto_data = get_text_proto_data(question_label)

  # convert it into proto format
  proto_data = sight_pb2.DecisionConfigurationStart()
  text_format.Parse(text_proto_data, proto_data)

  api_description = proto_data.choice_config[question_label].llm_config.description


  # Extract only action_attrs
  action_attrs = proto_data.action_attrs

  items = []
  for k, v_obj in action_attrs.items():
    # Access the description directly from the object and clean it
    description_text = v_obj.description.strip().rstrip('.')
    items.append(f"{k} : {description_text}")

  # Join all items with a comma and space, add a period at the end if not empty
  argument_description = ", \n".join(items) + ("." if items else "")

  return api_description, argument_description



def create_choice_config(
    label,
    description
) -> dict[str, sight_pb2.DecisionConfigurationStart.ChoiceConfig]:
  choice_config_dict = {}
  choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig()

  llm_config = sight_pb2.DecisionConfigurationStart.LLMConfig()
  llm_config.description = description
  choice_config.llm_config.CopyFrom(llm_config)

  choice_config_dict[label] = choice_config
  return choice_config_dict
