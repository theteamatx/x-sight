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

import dotenv
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sight.demo.agentic_demo.proposal_addition import addition_api_with_sight
from sight.demo.agentic_demo.proposal_division import division_api_with_sight
from sight.demo.agentic_demo.proposal_multiplication import multiplication_api_with_sight
from sight.demo.agentic_demo.proposal_subtraction import subtraction_api_with_sight


find_dotenv = dotenv.find_dotenv
load_dotenv = dotenv.load_dotenv

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

tools = [
    addition_api_with_sight, subtraction_api_with_sight,
    multiplication_api_with_sight, division_api_with_sight
]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

user_input = input(
    "Enter the operations you want to perform in plain english: "
)
response = agent.invoke({"input": user_input})
print("Response : ", response)
