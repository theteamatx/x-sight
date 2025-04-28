from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_demo.bayes_opt_worker import run_BO_tool
load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools([run_BO_tool])

# result = llm.invoke("Run a Bayesian optimization on the sphere function and tell me the best values of the input actions and the maximum reward it can achieve")
# result = llm_with_tools.invoke("Run a Bayesian optimization on the sphere function and tell me the best values of the input actions and the maximum reward it can achieve")
# print(result)

# Set up the agent
agent = initialize_agent(
    tools=[run_BO_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

response = agent.invoke("Run a Bayesian optimization on the sphere function and tell me the best values of the input actions and the maximum reward it can achieve")
print(response)
