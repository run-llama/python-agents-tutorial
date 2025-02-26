from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.tools.tavily_research import TavilyToolSpec
import os
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

llm = OpenAI(model="gpt-4o-mini")

tavily_tool = TavilyToolSpec( api_key=os.getenv("TAVILY_API_KEY") )

workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information."
)

async def main():
    handler = workflow.run(user_msg="What's the weather like in San Francisco?")

    # handle streaming output
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output            

    # print final output
    print(str(await handler))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
