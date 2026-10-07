"""
Logging utility for TERRA SAFE AI
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from src.utils.config import get_config


def setup_logger(name: str = "terra_safe_ai", level: Optional[str] = None) -> logging.Logger:
    """Setup and return a logger instance"""
    config = get_config()
    
    log_level = level or config.logging.level
    log_file = config.logging.file
    log_format = config.logging.format
    max_bytes = config.logging.max_bytes
    backup_count = config.logging.backup_count
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "terra_safe_ai") -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)