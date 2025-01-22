"""Logging configuration"""

import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure file handler
file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
))

# Setup app logger
app_logger = logging.getLogger('app')
app_logger.addHandler(file_handler)
app_logger.addHandler(console_handler)

def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    app_logger.setLevel(level)