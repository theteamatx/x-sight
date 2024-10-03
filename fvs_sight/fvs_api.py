from typing import Any

from sight.proto import sight_pb2

FVS_PARAMS = {
    "fire-SIMFIRE_{index}-6_stand_area_burned": None,
    "fire-SIMFIRE_{index}-1_cycle": None,
    "base-FERTILIZ-howManyCycle": None,
    "base-FERTILIZ-extra_step": None,
    "base-FERTILIZ-extra_offset": None,
}

MAX_FVS_CYCLE = 40


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
  return {
      key: (sight_pb2.DecisionConfigurationStart.AttrProps() if value
            is not None else sight_pb2.DecisionConfigurationStart.AttrProps()
           ) for key, value in config_dict.items()
  }


def expand_params_for_cycles(fvs_params: dict[str, Any]) -> dict[str, Any]:
  """Expand the params for cycles if {index} found in the key for all FVS_CYCLE.

  Args:
      fvs_params (dict[str, Any]): FVS params to be expanded which may contain
        {index} in the key.
  Returns:
      dict[str, Any]: Expanded params for all cycles.
  """
  new_params = {}
  for key, value in fvs_params.items():
    if '{index}' in key:
      for i in range(MAX_FVS_CYCLE):
        new_key = key.replace('{index}', str(i))
        new_params[new_key] = value
    else:
      new_params[key] = value
  return new_params


def get_action_attrs():
  """Returns the action attributes for the FVS action.
  """
  action_config = {'region': None, 'project_id': None}
  action_config.update(expand_params_for_cycles(fvs_params=FVS_PARAMS))
  return create_attr_props(action_config)


# action_attrs = {
#     "a1":
#     sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=0,
#         max_value=1,
#     ),
#     "a2":
#     sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=0,
#         max_value=1,
#     ),
#     "a3":
#     sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=0,
#         max_value=1,
#     ),
# }


def get_outcome_attrs():
  """Returns the outcome attributes for the FVS outcome.
  """
  outcome_config = {'time_series': None}
  return create_attr_props(outcome_config)


# outcome_attrs = {
#     "time_series":
#     sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=0,
#         max_value=1,
#     ),
# }
