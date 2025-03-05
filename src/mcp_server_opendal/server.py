import argparse
import base64
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server_opendal.resource import OPENDAL_OPTIONS, OpendalResource, parse_uri

load_dotenv()
default_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, default_log_level, logging.INFO)

# Initialize the FastMCP server - Use the default log level
mcp = FastMCP("opendal_service", log_level=default_log_level)

# Configure logging - still use numeric constants to configure standard logging
logging.basicConfig(level=log_level)
logger = logging.getLogger("mcp_server_opendal")


def register_resources():
    """Register all OpenDAL resources"""
    # Get all available schemes
    schemes = set([k.split("_")[0] for k in OPENDAL_OPTIONS.keys()])

    # Clean existing resources and register new resources
    for scheme in schemes:
        try:
            resource = OpendalResource(scheme)
            mcp.add_resource(resource)
            logger.info(f"Registered OpenDAL resource for scheme: {scheme}")
        except Exception as e:
            logger.error(
                f"Failed to register OpenDAL resource for scheme {scheme}: {e}"
            )


# Register resources
register_resources()


# Create a resource template, used to dynamically generate resources
@mcp.resource("{scheme}://{path}")
async def opendal_resource(scheme: str, path: str) -> Dict[str, Any]:
    """
    Access files in OpenDAL service

    Args:
        scheme: storage service scheme
        path: file path

    Returns:
        Dictionary containing file content and metadata
    """
    logger.debug(f"Reading template resource content: {scheme}://{path}")
    try:
        resource = OpendalResource(scheme)
        data = await resource.read_path(path)
        metadata = await resource.stat(path)

        if resource.is_text_file(path):
            return {
                "content": data.decode("utf-8"),
                "mime_type": metadata.content_type or "text/plain",
                "size": metadata.content_length,
                "is_binary": False,
            }
        else:
            return {
                "content": base64.b64encode(data).decode("ascii"),
                "mime_type": metadata.content_type or "application/octet-stream",
                "size": metadata.content_length,
                "is_binary": True,
            }
    except Exception as e:
        logger.error(f"Failed to read resource: {e!s}")
        return {"error": str(e)}


# Modify the list tool to ensure the path ends with a slash
@mcp.tool()
async def list(uri: str) -> str:
    """
    List files in OpenDAL service

    Args:
        uri: resource URI, e.g. mys3://path/to/dir

    Returns:
        String containing directory content
    """
    logger.debug(f"Listing directory content: {uri}")
    try:
        resource, path = parse_uri(uri)

        # Ensure directory path ends with a slash
        if path and not path.endswith("/"):
            path = path + "/"

        entries = await resource.list(path)

        return str(entries)
    except Exception as e:
        logger.error(f"Failed to list directory content: {e!s}")
        return f"Error: {e!s}"


# Read content of file
@mcp.tool()
async def read(uri: str) -> Dict[str, Any]:
    """
    Read file content from OpenDAL service

    Args:
        uri: resource URI, e.g. mys3://path/to/file

    Returns:
        File content or error information
    """
    logger.debug(f"Reading file content: {uri}")
    try:
        resource, path = parse_uri(uri)
        # Directly call the resource function to get content
        return await opendal_resource(resource.scheme, path)
    except Exception as e:
        logger.error(f"Failed to read file content: {e!s}")
        return {"error": str(e)}


# Get file metadata
@mcp.tool()
async def get_info(uri: str) -> str:
    """
    Get metadata of file in OpenDAL service

    Args:
        uri: resource URI, e.g. mys3://path/to/file

    Returns:
        File metadata information
    """
    logger.debug(f"Getting file info: {uri}")
    try:
        resource, path = parse_uri(uri)
        metadata = await resource.stat(path)

        result = f"File: {path}\n"
        result += f"Size: {metadata.content_length} bytes\n"
        result += f"Type: {metadata.content_type}\n"

        return result
    except Exception as e:
        logger.error(f"Failed to get file info: {e!s}")
        return f"Error: {e!s}"


def main():
    parser = argparse.ArgumentParser(description="OpenDAL MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport method (stdio or sse)",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run("sse")
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
