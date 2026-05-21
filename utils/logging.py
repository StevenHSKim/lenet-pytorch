import logging
import os


def get_logger(log_dir: str = "results", filename: str = "train.log"):
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, filename), mode="w")
        ]
    )
    return logging.getLogger()