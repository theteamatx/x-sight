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


@decision_worker(question_label=get_question_label())
def main(sight: Sight, action: dict) -> Tuple[float, dict]:
  mitigation_list = [
      227.6, 273.4, 273.3, 248.6, 165.3, 130.6, 106.4, 92.1, 81.7, 62.8
  ]
  sim_stream = pd.Series(mitigation_list)
  simulation_time = random.uniform(2, 2)

  return simulation_time, sim_stream


if __name__ == "__main__":
  # app.run(main)
  app.run(lambda _: sight.run_worker(main, {
      'label': get_question_label(),
  }))
