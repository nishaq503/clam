"""Helper functions and utilities for the package."""

import logging
import pathlib
import sys


def configure_logging(
    app_name: str,
    *,
    level: int = logging.INFO,
    fmt_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_path: pathlib.Path | None = None,
) -> logging.Logger:
    """Configures and returns a logger for the application.

    Args:
        app_name: The name of the application.
        level: The logging level.
        fmt_str: The logging format.
        file_path: Optional path to a file to log messages to.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt_str)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

__all__ = ["configure_logging"]
