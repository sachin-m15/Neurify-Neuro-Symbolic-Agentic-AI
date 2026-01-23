import logging

def get_logger(name: str = "verified_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
