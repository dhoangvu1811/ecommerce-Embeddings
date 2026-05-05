
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging():
    """
    Set up structured JSON logging.
    """
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    log_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])
    
    # Silence overly verbose loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
