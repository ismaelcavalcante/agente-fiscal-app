import logging
import os

def get_logger(name: str) -> logging.Logger:
    """
    Lightweight logger for Streamlit Cloud.
    Does not write to disk, only stdout.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Global logger instance
logger = get_logger("agente_fiscal")