import logging
import os
from config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes to both console and logs/app.log.
    
    Args:
        name: typically __name__ from the calling module
        
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler — DEBUG and above (everything)
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "app.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger