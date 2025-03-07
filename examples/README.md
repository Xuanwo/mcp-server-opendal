# LlamaIndex Agent Example with OpenDAL MCP

## Start the MCP Server

To run this example, you need to have a MCP server running.

Make sure you are in the root directory of the project, set the environment variables first:

- `OPENDAL_FS_TYPE=fs`
- `OPENDAL_FS_ROOT=./examples/`

Then, run the following command:

```bash
uv sync  # To install the project, this should be done only once
uv run mcp-server-opendal --transport sse
```

## Run the Example

Set the environment variables below.

- `MCP_HOST`: The host of the MCP server
- `MCP_PORT`: The port of the MCP server
- `OPENAI_API_KEY`: The API key of the OpenAI API
- `OPENAI_MODEL`: The model of the OpenAI API
- `OPENAI_ENDPOINT`: The endpoint of the OpenAI API

Then, run the example with the following command:

```bash
uv run examples/llamaindex-with-opendal-mcp.py
```
