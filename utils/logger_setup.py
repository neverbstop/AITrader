import logging

def setup_logging(log_file: str = None, level: int = logging.INFO):
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(level=level, format=log_format)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(handler)
    print("✅ Logging setup complete.")
