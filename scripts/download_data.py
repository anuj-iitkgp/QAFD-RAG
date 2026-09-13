"""
Script to download or prepare benchmark multi-hop QA datasets.
Prepares MuSiQue, HotpotQA, and 2WikiMultiHopQA data in data/raw/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from data.download import prepare_sample_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_data")


def main():
    logger.info("Initializing multi-hop QA datasets...")
    base_dir = "data/raw"
    os.makedirs(base_dir, exist_ok=True)
    paths = prepare_sample_datasets(base_dir)
    logger.info("All datasets successfully prepared:")
    for name, path in paths.items():
        logger.info(f"  - {name.upper()}: {path}")


if __name__ == "__main__":
    main()
