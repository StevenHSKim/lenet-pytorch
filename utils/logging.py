import logging
import os
from datetime import datetime


def get_logger(log_dir: str = "results", filename: str = "train.log"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = filename.replace(".log", f"_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, filename), mode="w")
        ]
    )
    return logging.getLogger()