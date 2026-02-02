import numpy as np

class BruteForceKNN:
    """
    Exact k-NN over in-memory vectors (list of list[float])
    """
    def __init__(self, X, metric="cosine"):
        self.X = np.array(X, dtype=np.float32)
        self.metric = metric.lower()

    def topk(self, query_vec, k=10):
        q = np.array(query_vec, dtype=np.float32)
        if self.metric == "cosine":
            sims = self.X @ q
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]
        elif self.metric == "euclidean":
            dists = np.linalg.norm(self.X - q, axis=1)
            idx = np.argsort(dists)[:k]
            scores = dists[idx]
        elif self.metric == "manhattan":
            dists = np.sum(np.abs(self.X - q), axis=1)
            idx = np.argsort(dists)[:k]
            scores = dists[idx]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        return idx, scores
