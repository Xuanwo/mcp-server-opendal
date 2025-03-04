import asyncio
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import logging
from mcp.types import (
    LoggingLevel,
    EmptyResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

import base64
from .resource import parse_uri
from pydantic import AnyUrl


# Initialize server
server = Server("opendal_service")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_server_opendal")


@server.set_logging_level()
async def set_logging_level(level: LoggingLevel) -> EmptyResult:
    logger.setLevel(level.lower())
    await server.request_context.session.send_log_message(
        level="info", data=f"Log level set to {level}", logger="mcp_server_opendal"
    )
    return EmptyResult()


# @server.list_resources()
# async def list_resources() -> List[Resource]:
#     logger.debug("Starting to list resources")
#     resources = []

#     schemes = set([k.split("_")[0] for k in OPENDAL_OPTIONS.keys()])
#     logger.debug(f"Schemes: {schemes}")

#     for scheme in schemes:
#         resource = OpendalResource(scheme)
#         resources.append(resource)

#     logger.info(f"Returning {len(resources)} resources")
#     return resources


# @server.read_resource()
# async def read_resource(uri: AnyUrl) -> str:
#     """
#     Read content from an opendal resource and return structured response

#     Returns:
#         Dict containing 'contents' list with uri, mimeType, and text for each resource
#     """
#     uri_str = str(uri)
#     logger.debug(f"Reading resource: {uri_str}")

#     resource, path = parse_uri(uri)

#     logger.debug(f"Attempting to read - scheme: {resource.scheme()}, path: {path}")

#     metadata = await resource.stat(path)
#     logger.debug(f"Path: {path} - Metadata: {metadata}")

#     data = await resource.read(path)

#     # Process the data based on file type
#     if resource.is_text_file(path):
#         result = ReadResourceResult(
#             contents=[
#                 TextResourceContents(
#                     text=data.decode("utf-8"),
#                     uri=uri_str,
#                     mimeType=metadata.content_type,
#                 )
#             ]
#         )
#     else:
#         result = ReadResourceResult(
#             contents=[
#                 BlobResourceContents(
#                     blob=str(base64.b64encode(data)),
#                     uri=uri_str,
#                     mimeType=metadata.content_type,
#                 )
#             ]
#         )

#     logger.debug(result)
#     return result


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="list",
            description="Returns some or all of the files in an opendal service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to list. For example, mys3://path/to/dir",
                    },
                },
                "required": ["uri"],
            },
        ),
        Tool(
            name="read",
            description="Reads the contents of a file from an opendal service.",
            inputSchema={
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to list. For example, mys3://path/to/file",
                    },
                },
                "required": ["uri"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[TextContent | ImageContent | EmbeddedResource]:
    logger.debug(f"Handling call tool: {name} with arguments: {arguments}")

    resource, path = parse_uri(AnyUrl(arguments["uri"]))

    try:
        match name:
            case "list":
                logger.debug(
                    f"Attempting to list - scheme: {resource.scheme}, path: {path}"
                )

                files = await resource.list(path)
                return [TextContent(type="text", text=str(files))]
            case "read":
                logger.debug(
                    f"Attempting to read - scheme: {resource.scheme}, path: {path}"
                )

                data = await resource.read(path)
                # Process the data based on file type
                if resource.is_text_file(path):
                    file_content = data.decode("utf-8")
                else:
                    file_content = str(base64.b64encode(data))

                return [TextContent(type="text", text=str(file_content))]
    except Exception as error:
        return [TextContent(type="text", text=f"Error: {str(error)}")]


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-server-opendal",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
