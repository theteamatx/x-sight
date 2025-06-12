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
"""Demo of using Optimizer tool with Langchain."""

from typing import Any, Dict, Sequence

from absl import app
from absl import flags
import dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sight.tools.proposal_tool import proposal_api
from sight.sight import Sight
from sight.tools.tool_helper import create_lc_tool
from sight.widgets.decision import decision

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from helpers.logs.logs_handler import logger as logging

from sight.demo.agentic_demo.python_parser import PythonCodeParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate

from sight_service.proto import service_pb2
from sight import service_utils as service

load_dotenv = dotenv.load_dotenv
find_dotenv = dotenv.find_dotenv
load_dotenv(find_dotenv())
FLAGS = flags.FLAGS

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20",)
code_parser = PythonCodeParser()


def get_question_label():
  return "Pyrolyzer"


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the sight related config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "Pyrolyzer_label"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight:

    # user_input = input(
    #     "Enter the operations you want to perform in plain english: "
    # )

    system_input = """You are an expert Python programmer.
            Your sole task is to write a complete, working Python function based on the user's request.
            The function should be well-commented, follow best practices, and be ready to use.
            Do NOT include any explanations, introductory text, or concluding remarks.
            """

    user_input = (
        "Generate a python function that takes input as X, Y and Z as "
        "key of single action_params dictionary and calculate the "
        "reward that maximize the Y by minimizing the X and Z**2"
        "output of the function should be tuple with first element as"
        "calculated reward and another element as outcome which is "
        "dictionary with key final_result and value as reward")

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_input),
         HumanMessage(content=user_input)])

    # chain = prompt | llm
    chain = prompt | llm | code_parser

    generated_python_function_string = chain.invoke({})
    print("Response: ", generated_python_function_string)

    # # Sending generated function to sight
    # req = service_pb2.SendFunctionRequest()
    # req.client_id = str(sight.id)
    # req.question_label = get_question_label()
    # req.function_code = generated_python_function_string

    # response = service.call(
    #     lambda s, meta: s.SendFunction(req, 300, metadata=meta))
    # logging.info('##### SendFunction response=%s #####', response)


if __name__ == "__main__":
  app.run(main)
