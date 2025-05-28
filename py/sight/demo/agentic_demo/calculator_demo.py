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
"""Demo of using calculator tool with Langchain."""

from typing import Any, Dict, Sequence

from absl import app
from absl import flags
import dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from sight.demo.agentic_demo.tools.calculator_tool import calculator_api
from sight.demo.agentic_demo.tools.calculator_tool import generate_description
from sight.sight import Sight
from sight.widgets.decision import decision


load_dotenv = dotenv.load_dotenv
find_dotenv = dotenv.find_dotenv
load_dotenv(find_dotenv())
FLAGS = flags.FLAGS

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the sight related config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "multiple_opt_label"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight_instance:

    # function to be used by tool - wrapped with sight object
    def calculator_tool_fn(action_dict: Dict[str, Any]):
      return calculator_api(action_dict=action_dict, sight=sight_instance)

    # tool object to be used by agent
    calculator_tool = StructuredTool.from_function(
        name="calculator_tool",
        func=calculator_tool_fn,
        verbose=True,
        description=generate_description(),
    )
    tools = [calculator_tool]

    # initialize agent with tools and llm
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    user_input = input(
        "Enter the operations you want to perform in plain english: "
    )
    # user_input = "can you please divide 25 by 5"

    response = agent.invoke({"input": user_input})
    print("Response: ", response)


if __name__ == "__main__":
  app.run(main)
