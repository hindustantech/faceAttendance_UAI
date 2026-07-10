# app/utils/logger.py
import logging
import sys
import json
from datetime import datetime
from app.config import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Setup structured logger
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Create formatter
    if settings.LOG_FORMAT == "json":
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                if hasattr(record, 'extra'):
                    log_entry.update(record.extra)
                
                if record.exc_info and record.exc_info[1]:
                    log_entry["exception"] = str(record.exc_info[1])
                
                return json.dumps(log_entry)
        
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger