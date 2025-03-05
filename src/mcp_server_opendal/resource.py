import logging
import os
from typing import Any, List, Tuple, Union

import opendal
from dotenv import load_dotenv
from mcp.server.fastmcp.resources import Resource
from opendal import Entry, Metadata
from opendal.layers import RetryLayer
from pydantic import Field

logger = logging.getLogger("mcp_server_opendal")

# all opendal related environment variables
load_dotenv()
OPENDAL_OPTIONS = {
    k.replace("OPENDAL_", "").lower(): v.lower()
    for k, v in os.environ.items()
    if k.startswith("OPENDAL_")
}


class OpendalResource(Resource):
    """
    OpenDAL Resource provider that handles interactions with different storage services.
    Acts both as a FastMCP Resource and as an interface to OpenDAL operations.

    This resource provider will read the environment variables for the given scheme and use them to configure the opendal operator.

    For example, if the scheme is "mys3", the environment variables should be:

    ```
    OPENDAL_MYS3_TYPE=s3
    OPENDAL_MYS3_BUCKET=mybucket
    OPENDAL_MYS3_REGION=us-east-1
    OPENDAL_MYS3_ENDPOINT=http://localhost:9000
    OPENDAL_MYS3_ACCESS_KEY_ID=myaccesskey
    OPENDAL_MYS3_SECRET_ACCESS_KEY=mysecretkey
    ```
    """

    scheme: str = Field("", description="Storage scheme (e.g., s3, fs)")
    op: Any = None

    def __init__(self, scheme: str):
        scheme = scheme.lower()
        # Configure OpenDAL operator
        opendal_type = OPENDAL_OPTIONS.get(f"{scheme}_type")
        opendal_options = {
            k.replace(f"{scheme}_", ""): v
            for k, v in OPENDAL_OPTIONS.items()
            if k.startswith(f"{scheme}_")
        }
        logger.debug(f"Initializing OpendalResource with options: {opendal_options}")

        # Initialize FastMCP Resource
        super().__init__(
            uri=f"{scheme}://",
            name=f"{scheme} storage",
            description=f"Storage service accessed via OpenDAL {scheme} protocol",
            mime_type="application/vnd.folder",  # for containers/directories
        )

        # Initialize OpenDAL operator
        self.scheme = scheme
        self.op = opendal.AsyncOperator(opendal_type, **opendal_options).layer(
            RetryLayer()
        )
        logger.debug(f"Initialized OpendalResource: {self.op}")

    async def read(self) -> str:
        """
        Read method for scheme-level resources, only returns descriptive information
        Actual file content is obtained through path-specific methods
        """
        info = f"OpenDAL {self.scheme} storage resource.\n\n"
        info += f"To access specific files, use: {self.scheme}://path/to/file\n"
        info += f"To list directory contents, use the 'list' tool with: {self.scheme}://path/to/dir\n"
        return info

    async def list(
        self, prefix: Union[str, os.PathLike], max_keys: int = 1000
    ) -> List[Entry]:
        """List entries with the given prefix"""
        logger.debug(f"Listing entries with prefix: {prefix}")

        if max_keys <= 0:
            return []

        entries = []

        it = await self.op.list(prefix)

        async for entry in it:
            logger.debug(f"Listing entry: {entry}")
            entries.append(entry)
            if len(entries) >= max_keys:
                break

        return entries

    async def read_path(self, path: Union[str, os.PathLike]) -> bytes:
        """Read content from a specific path"""
        logger.debug(f"Reading path: {path}")
        return await self.op.read(path)

    async def stat(self, path: Union[str, os.PathLike]) -> Metadata:
        """Get metadata for a specific path"""
        logger.debug(f"Statting path: {path}")
        return await self.op.stat(path)

    def is_text_file(self, path: Union[str, os.PathLike]) -> bool:
        """Determine if a file is text-based by its extension"""
        text_extensions = {
            ".txt",
            ".log",
            ".json",
            ".xml",
            ".yml",
            ".yaml",
            ".md",
            ".csv",
            ".ini",
            ".conf",
            ".py",
            ".js",
            ".html",
            ".css",
            ".sh",
            ".bash",
            ".cfg",
            ".properties",
        }
        return any(path.lower().endswith(ext) for ext in text_extensions)


def parse_uri(uri: str) -> Tuple[OpendalResource, str]:
    """Parse a URI into a resource and path"""
    from urllib.parse import unquote, urlparse

    logger.debug(f"Parsing URI: {uri}")
    parsed = urlparse(uri)

    scheme = parsed.scheme
    path = parsed.netloc + parsed.path
    path = unquote(path)  # Decode URL-encoded characters
    return (OpendalResource(scheme), path)
