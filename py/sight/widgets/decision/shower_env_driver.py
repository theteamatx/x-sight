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
"""Demo of Drivier function to be used in case Sight used without any environment."""

from helpers.logs.logs_handler import logger as logging
import random
import numpy as np
from sight import data_structures
from sight.sight import Sight
from sight.widgets.decision import decision

_file_name = "shower_env_driver.py"


def driver_fn(sight: Sight) -> None:
    """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
    method_name = 'driver_fn'
    logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

    logging.info('sight.widget_decision_state : %s',
                 sight.widget_decision_state)

    temperature = 38 + random.randint(-3, 3)
    shower_length = 60
    data_structures.log_var("Temperature", temperature, sight)

    for _ in range(shower_length):
        # Ask Sight's optimizer for the action to perform.
        chosen_action = decision.decision_point("DP_label", sight)
        # direction = np.array(chosen_action["Direction"], dtype=np.int64)

        # Change temperature based on the Sight-recommended direction.
        temperature += chosen_action["Direction"]
        logging.info('temperature=%s, direction=%s', temperature,
                     chosen_action["Direction"])
        data_structures.log_var("Temperature", temperature, sight)

        # Calculate reward based on whether the temperature target has
        # been achieved.
        if temperature >= 37 and temperature <= 39:
            current_reward = 1
        else:
            current_reward = -abs(temperature - 38)

        # Inform Sight of the outcome of the recommended action.
        decision.decision_outcome(
            "DO_label",
            current_reward,
            sight,
        )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
