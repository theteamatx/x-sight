from typing import Any

from sight.worker.worker_helper import create_attr_props



FVS_PARAMS = {
    "fire-SIMFIRE_{index}-6_stand_area_burned": None,
    "fire-SIMFIRE_{index}-1_cycle": None,
    "base-FERTILIZ-howManyCycle": None,
    "base-FERTILIZ-extra_step": None,
    "base-FERTILIZ-extra_offset": None,
}

MAX_FVS_CYCLE = 40


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
  # action_config.update(expand_params_for_cycles(fvs_params=FVS_PARAMS))
  return create_attr_props(action_config)


def get_outcome_attrs():
  """Returns the outcome attributes for the FVS outcome.
  """
  outcome_config = {'time_series': None}
  return create_attr_props(outcome_config)
