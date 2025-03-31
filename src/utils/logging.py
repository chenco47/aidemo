"""
Logging utilities for the project.
"""
import logging
import os
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Configure and return a logger.
    
    Args:
        name: The name of the logger (typically __name__)
        level: The log level (if None, uses LOG_LEVEL env var or defaults to INFO)
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or use INFO as default
    if level is None:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger 