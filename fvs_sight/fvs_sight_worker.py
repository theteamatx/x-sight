import json
import os
import random
import time
from typing import Sequence

from absl import app
from absl import flags
# from fvs_sight.fvs_api import action_attrs, outcome_attrs
from fvs_sight import fvs_api
import pandas as pd
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import decision_episode_fn
import yaml


def get_question_label():
  return 'Q_label3'

def simulate_fvs(sight, params_dict):
  # print('here params_dict is :', params_dict)
  mitigation_list = [
      227.6, 273.4, 273.3, 248.6, 165.3, 130.6, 106.4, 92.1, 81.7, 62.8
  ]
  sim_stream = pd.Series(mitigation_list)
  simulation_time = 10
  time.sleep(simulation_time)
  print(f'sleeping for some time he he 😪 , {simulation_time}')
  # print(sim_stream)
  return sim_stream


def driver_fn(sight):

  params_dict = decision.decision_point(get_question_label(), sight)
  # params_dict = {'fvs_type':'managed','region':'BM','project_id':'ACR173','desc': 'fire_projectACR173', 'fire-SIMFIRE_27-1_cycle': 2028, 'fire-SIMFIRE_27-6_stand_area_burned': 10.0, 'fire-SIMFIRE_30-1_cycle': 2031, 'fire-SIMFIRE_30-6_stand_area_burned': 10.0, 'fire-SIMFIRE_31-1_cycle': 2032, 'fire-SIMFIRE_31-6_stand_area_burned': 10.0}
  print('params_dict : ', params_dict)
  # if(params_dict == None):
  #    return None
  # raise SystemError

  sim_stream = simulate_fvs(sight, params_dict)

  outcome = {'time_series': sim_stream}
  print("outcome : ", outcome)

  decision.decision_outcome('outcome_label', sight, reward=0, outcome=outcome)
  # return sight


#temporary created
def get_sight_instance():
  params = sight_pb2.Params(
      label="kokua_experiment",
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(driver_fn=driver_fn,
                 sight=sight,
                 action_attrs=fvs_api.get_action_attrs(),
                 outcome_attrs=fvs_api.get_outcome_attrs(),
                 question_label=get_question_label)


if __name__ == "__main__":
  app.run(main)
