from langchain.agents import AgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish

from langchain.schema import OutputParserException
import re

class CustomPythonCodeParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        code_match = re.search(r"```python\n(.*?)```", llm_output, re.DOTALL)
        if code_match:
            cleaned_code = code_match.group(1).strip()
            return AgentFinish(
                return_values={"output": cleaned_code},
                log=llm_output,
            )
        final_answer_match = re.search(r"Final Answer:\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return AgentFinish(
                return_values={"output": final_answer},
                log=llm_output,
            )

        action_match = re.search(r"Action:\s*(.*)\nAction Input:\s*(.*)", llm_output, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


        raise OutputParserException(f"Could not parse LLM output: {llm_output}")

    def get_format_instructions(self) -> str:
        return """Your output should be a complete, working Python function. If you include any markdown fences (```python), please ensure they completely enclose the code. If you are done, just output the raw Python code. If you need to use a tool, format your response with "Action: <tool_name>\nAction Input: <tool_input>". If you are done, output "Final Answer: <your_answer>" or just the raw Python code for the function."""
