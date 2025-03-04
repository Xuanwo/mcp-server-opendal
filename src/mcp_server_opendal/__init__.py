from . import server
import asyncio
import logging


def main():
    """Main entry point for the package."""
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(server.main())


# Optionally expose other important items at package level
__all__ = ["main", "server"]
