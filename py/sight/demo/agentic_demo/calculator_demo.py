from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_tool_calling_agent, AgentExecutor

from sight.demo.agentic_demo.proposal_calculator import calculator_api_with_sight
load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# llm_with_tools = llm.bind_tools([calculator_tool])

# Set up the agent
tools = [calculator_api_with_sight]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

user_input = input("Enter the operations you want to perform in plain english: ")
response = agent.invoke({"input": user_input})
print(response)
