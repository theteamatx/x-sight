# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demo of using the Sight Decision API to run forest simulator."""

import os
import random
from typing import Sequence

from absl import app
from absl import flags
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FLAGS = flags.FLAGS

# Prepare the data.
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the black box function to optimize.
def black_box_function(C, degree):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C=C, degree=degree)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    f = roc_auc_score(y_test, y_score)
    return f


def driver(sight: Sight) -> None:
    """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """

    for _ in range(1):
        next_point = decision.decision_point("label", sight)
        next_point["degree"] = int(next_point["degree"])
        reward = black_box_function(next_point["C"], next_point["degree"])

        sight.text("C=%s, degree=%s, f(x)=%s" % (
            next_point["C"],
            next_point["degree"],
            reward,
        ))
        print("C : ", next_point["C"], ", degree : ", next_point["degree"], ", reward : ", reward)
        decision.decision_outcome("target", reward, sight)


def get_sight_instance():
    params = sight_pb2.Params(
        label='kokua_experiment',
        bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )
    sight_obj = Sight(params)
    return sight_obj


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with get_sight_instance() as sight:
        decision.run(
            driver_fn=driver,
            action_attrs={
                "C":
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=0.1,
                    max_value=10,
                ),
                "degree":
                sight_pb2.DecisionConfigurationStart.AttrProps(
                    min_value=1,
                    max_value=5,
                ),
            },
            sight=sight,
        )


if __name__ == "__main__":
    app.run(main)
