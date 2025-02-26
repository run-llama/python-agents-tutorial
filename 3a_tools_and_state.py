from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.workflow import Context

llm = OpenAI(model="gpt-4o-mini")

async def set_name(ctx: Context, name: str) -> str:
    state = await ctx.get("state")
    state["name"] = name
    await ctx.set("state", state)
    return f"Name set to {name}"

workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},
)

async def main():
    ctx = Context(workflow)

    # check if it knows a name before setting it
    response = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(str(response))

    # set the name using a tool
    response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
    print(str(response2))

    # retrieve the value from the state directly
    state = await ctx.get("state")
    print("Name as stored in state: ",state["name"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
