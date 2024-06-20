from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

# settings
Settings.llm = OpenAI(model="gpt-4o",temperature=0)

# function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply_tool, add_tool])

agent = ReActAgent.from_tools(finance_tools, verbose=True)

response = agent.chat("What is the current price of NVDA?")

print(response)
