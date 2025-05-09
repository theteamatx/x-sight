import os
import random
import time
from typing import Sequence, Tuple

from absl import app
import pandas as pd
from sight.proto import sight_pb2
from sight.sight import Sight
from sight import sight
from sight.widgets.decision import decision
from helpers.decorators.decision_worker import decision_worker



def get_question_label():
  return 'FVS'


@decision_worker(question_label = get_question_label())
def main(sight: Sight, action: dict) -> Tuple[float, dict]:
  # print('here params_dict is :', params_dict)
  mitigation_list = [
      227.6, 273.4, 273.3, 248.6, 165.3, 130.6, 106.4, 92.1, 81.7, 62.8
  ]
  sim_stream = pd.Series(mitigation_list)
  simulation_time = random.uniform(2, 2)

  return simulation_time, sim_stream


# def driver_fn(sight):

#   params_dict = decision.decision_point(get_question_label(), sight)
#   # params_dict = {'fvs_type':'managed','region':'BM','project_id':'ACR173','desc': 'fire_projectACR173', 'fire-SIMFIRE_27-1_cycle': 2028, 'fire-SIMFIRE_27-6_stand_area_burned': 10.0, 'fire-SIMFIRE_30-1_cycle': 2031, 'fire-SIMFIRE_30-6_stand_area_burned': 10.0, 'fire-SIMFIRE_31-1_cycle': 2032, 'fire-SIMFIRE_31-6_stand_area_burned': 10.0}
#   print('params_dict : ', params_dict)
#   # if(params_dict == None):
#   #    return None
#   # raise SystemError

#   sim_stream = simulate_fvs(sight, params_dict)

#   outcome = {'time_series': sim_stream}
#   print("outcome : ", outcome)

#   decision.decision_outcome('outcome_label', sight, reward=0, outcome=outcome)
#   # return sight


# def main(argv: Sequence[str]) -> None:
#   if len(argv) > 1:
#     raise app.UsageError("Too many command-line arguments.")

#   # Enry point for the worker to start asking for the FVS related actions
#   sight.run_worker(question_label=get_question_label(), driver_fn=driver_fn)


if __name__ == "__main__":
  # app.run(main)
  app.run(lambda _ : sight.run_worker(main, get_question_label()))
