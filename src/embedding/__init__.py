"""
Lightweight loader for precomputed Caltech101 embeddings.
"""

from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2] / "data"
EMB_PATH = BASE_DIR / "caltech101_embeddings_norm.npy"
IDS_PATH = BASE_DIR / "caltech101_image_ids.npy"


def load_embeddings():
    """
    Returns:
        X: np.ndarray of shape (N, 512) float32, already L2-normalized
        image_ids: np.ndarray of relative paths (strings) into the dataset
    """
    X = np.load(EMB_PATH).astype(np.float32)
    image_ids = np.load(IDS_PATH, allow_pickle=True)
    return X, image_ids
