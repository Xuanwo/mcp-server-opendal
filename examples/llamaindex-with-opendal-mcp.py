import asyncio
import os

from llama_index.tools.mcp import McpToolSpec
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent import ReActAgent, ReActChatFormatter
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from llama_index.tools.mcp import McpToolSpec, BasicMCPClient
from dotenv import load_dotenv

load_dotenv()

# To run this example, you need to have a MCP server running.
# You can start a MCP server by running `mcp-server-opendal` in the root directory.
#
# And set the environment variables first.
# - OPENDAL_FS_TYPE=fs
# - OPENDAL_FS_ROOT=./examples/
#
# Then, run the following command:
#
# ```bash
# python ./src/mcp_server_opendal/server.py --transport sse
# ```
#
# And set the environment variables below.
# - MCP_HOST: The host of the MCP server
# - MCP_PORT: The port of the MCP server
# - OPENAI_API_KEY: The API key of the OpenAI API
# - OPENAI_MODEL: The model of the OpenAI API
# - OPENAI_ENDPOINT: The endpoint of the OpenAI API
#
# Then, run the following command:
#
# ```bash
# python examples/llamaindex-with-opendal-mcp.py
# ```

# MCP Server Connection Parameters
MCP_HOST = os.getenv("MCP_HOST")
MCP_PORT = os.getenv("MCP_PORT")

SYSTEM_PROMPT = """\
You are an agricultural research assistant. I am researching the impact of climate change on crop yields.
I need you to help me analyze some research data stored in the file system.

Please first list the files in the `fs://data/` directory,
Then you must read the file `README.txt` to understand the data,
then read the necessary CSV files, and summarize the main findings.
"""


async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = ReActAgent.from_tools(
        llm=OpenAILike(model=os.getenv("OPENAI_MODEL"), api_base=os.getenv("OPENAI_ENDPOINT"), api_key=os.getenv("OPENAI_API_KEY"), is_chat_model=True),
        tools=list(tools),
        react_chat_formatter=ReActChatFormatter(
            system_header=SYSTEM_PROMPT + "\n" + REACT_CHAT_SYSTEM_HEADER,
        ),
        max_iterations=20,
        verbose=True,
    )
    return agent


async def handle_user_message(message_content: str, agent: ReActAgent):
    user_message = ChatMessage.from_str(role="user", content=message_content)
    response = await agent.achat(message=user_message.content)
    print(response.response)

async def main():
    mcp_tool = McpToolSpec(client=BasicMCPClient(f"http://{MCP_HOST}:{MCP_PORT}/sse"))

    agent = await get_agent(mcp_tool)
    try:
        await handle_user_message("What is the main finding of the data?", agent)
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


if __name__ == "__main__":
    asyncio.run(main())
