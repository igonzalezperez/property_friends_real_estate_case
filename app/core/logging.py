"""
Custom Logging Handler for Loguru Integration
"""

import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Custom logging handler for Loguru integration with Python's logging module.
    Overrides the emit method of logging.Handler to send log records to Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit log records to Loguru logger.

        :param record: The log record to emit.
        """
        logger_opt = logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())
