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
logger = logging.getLogger("agente_fiscal")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

def log_state(state, title="STATE"):
    logger.debug(f"🔎 {title}: {state}")