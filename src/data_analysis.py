import os
import datasets
from datasets import load_dataset
from pathlib import Path
import numpy as np

os.makedirs("data", exist_ok=True)
datasets.config.DOWNLOADED_DATASETS_PATH = Path("data")

data = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
print(len(data))
print(data[0])
