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
"""Demo of using calculator tool with multiple worker using Langchain."""

from typing import Sequence

from absl import app
from absl import flags
import dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sight.sight import Sight
from sight.tools.tool_helper import create_lc_tool
from sight.widgets.decision import decision


find_dotenv = dotenv.find_dotenv
load_dotenv = dotenv.load_dotenv
load_dotenv(find_dotenv())
FLAGS = flags.FLAGS

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the sight related config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "Addition_label"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight:

    # creating langchain tools with propose_action_api as default function
    tools = [
        create_lc_tool("Addition", sight),
        create_lc_tool("Subtraction", sight),
        create_lc_tool("Multiplication", sight),
        create_lc_tool("Division", sight),
    ]

    # initialize agent with tools and llm
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    user_input = input(
        "Enter the operations you want to perform in plain english: ")
    # user_input = ("can you please add 25 by 5 and multiply it by 3 and then "
    #               "divide it by 2")

    response = agent.invoke({"input": user_input})
    print("Response: ", response)


if __name__ == "__main__":
  app.run(main)
