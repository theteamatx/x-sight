import json
import os
import random
import time
from typing import Sequence

from absl import app
from absl import flags
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision


def get_question_label():
  return 'Q_label4'


def calculator(v1, v2, ops):
  if (ops.lower() == "addition"):
    return v1 + v2
  elif (ops.lower() == "subtraction"):
    return v1 - v2
  elif (ops.lower() == "multiplication"):
    return v1 * v2
  elif (ops.lower() == "division"):
    return v1 / v2
  else:
    return "not supported operation by this calculator"


def driver_fn(sight):

  params_dict = decision.decision_point(get_question_label(), sight)
  print("params_dict here is : ", params_dict)

  result = calculator(params_dict["v1"], params_dict["v2"], params_dict["ops"])

  outcome = {'result': result}
  print("outcome : ", outcome)
  decision.decision_outcome('outcome_label', sight, reward=0, outcome=outcome)


def get_sight_instance():
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(
        driver_fn=driver_fn,
        sight=sight,
        question_label=get_question_label())


if __name__ == "__main__":
  app.run(main)
