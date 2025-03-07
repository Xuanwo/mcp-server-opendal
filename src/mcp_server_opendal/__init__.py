import logging

from . import server


def main():
    """Main entry point for the package."""
    logging.basicConfig(level=logging.DEBUG)
    server.main()


# Optionally expose other important items at package level
__all__ = ["main", "server"]
