from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2] / "data"
EMB_PATH = BASE_DIR / "caltech101_embeddings_norm.npy"
IDS_PATH = BASE_DIR / "caltech101_image_ids.npy"


def load_embeddings():

    X = np.load(EMB_PATH).astype(np.float32)
    image_ids = np.load(IDS_PATH, allow_pickle=True)
    return X, image_ids
