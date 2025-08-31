from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

# ---- Tools & LLM ----
tools = [TavilySearch()]
llm = ChatOllama(model="llama3", temperature=0)

# ---- Output parser with auto-fixing ----
base_parser = PydanticOutputParser(pydantic_object=AgentResponse)
output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# ---- Prompt with format instructions ----
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=base_parser.get_format_instructions())

# ---- Agent ----
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---- Chain: run agent → take final answer → parse JSON (with fixing) ----
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the Gurgaon on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
