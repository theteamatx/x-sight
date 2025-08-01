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
"""Demo of spawning multiple worker which can interact with each other."""

from typing import Sequence
import warnings
import asyncio
import inspect
from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from sight.demo.agentic_demo.tool_python_code_validator import validate_python_code


def warn(*args, **kwargs):
  pass


warnings.warn = warn

FLAGS = flags.FLAGS


def reward_fn(outcome):
  outcome_timeseries = outcome['time_series']
  return sum(outcome_timeseries) + 111


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "multiple_opt_label"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight:

    logging.info("spawned the workers.................")

    # initialize agent with tools and llm
    agent = initialize_agent(
        tools=[validate_python_code],  #tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # user_input = (
    #     "Generate a python function that takes input as list of carbon offset "
    #     "and calculate the reward by summing all the items in the list."
    #     "output of the function should be reward value of float type"
    #     "Finally, report the result of the validation as your final answer."
    # )

    user_input = (
        "1. Generate a Python function that takes a list of carbon offsets and "
        "calculates the reward by summing all the items. KEEP function name as "
        "reward_fn \n"

        # "IMPORTANT: In the generated function, you MUST intentionally "
        # "introduce a syntax error. For example, forget the colon ':' "
        # "after the function definition 'def function_name(args)'."
        "2. After generating the code, you MUST use the `validate_python_code` tool to verify it.\n"
        "3. **If the validation tool returns an error**, you MUST analyze the error, "
        "fix the Python code, and call the `validate_python_code` tool again on the "
        "corrected code. Repeat this process until the validation is successful.\n"
        "4. Once the validation is successful, provide only the final, correct Python function as your answer."
    )

    response = agent.invoke([
        SystemMessage(content="""You are an expert Python programmer.
            Your sole task is to write a complete, working Python function based on the user's request.
            The function should be well-commented, follow best practices, and be ready to use.
            Do NOT include any import statements, explanations, introductory text, or concluding remarks.
            Just plain, raw Python function. After generating the function, you MUST use the validate_python_code"
            tool to verify it.
        """),
        HumanMessage(content=user_input)
    ])
    print("Response: ", response['output'])

    actions = {
        "question_label_to_propose": "Fvs",
        "num_questions": 6,
        "batch_size": 3,
        "random_seed": 0,
        # "reward_fn_str": inspect.getsource(reward_fn)
        "reward_fn_str": response['output']
    }
    asyncio.run(
        proposal.propose_actions(
            sight=sight,
            question_label='Optimize',
            action_dict=actions,
        ))


if __name__ == "__main__":
  app.run(main)
