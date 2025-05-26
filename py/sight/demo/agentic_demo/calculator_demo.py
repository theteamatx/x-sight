from typing import Any, Sequence, Dict
from absl import app, flags

from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import Tool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType

from sight.demo.agentic_demo.tools.calculator_tool import generate_description, calculator_api
from sight.sight import Sight
from sight.widgets.decision import decision
from functools import partial
from langchain.globals import set_debug

FLAGS = flags.FLAGS
load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {'label': 'multiple_opt_label'}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config) as sight_instance:

    # set_debug(True)
    def calculator_tool_fn(action_dict: Dict[str, Any]):
      return calculator_api(action_dict=action_dict, sight=sight_instance)

    calculator_tool = StructuredTool.from_function(
        name="calculator_tool",
        func=calculator_tool_fn,
        verbose=True,
        description=generate_description(),
    )
    tools = [calculator_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # user_input = input("Enter the operations you want to perform in plain english: ")
    user_input = "can you please divide 25 by 5"
    response = agent.invoke({"input": user_input})
    print(response)


if __name__ == "__main__":
  app.run(main)
