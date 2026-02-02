import numpy as np
from collections import defaultdict
from typing import List, Set

class RandomHyperplaneLSH:

    def __init__(self, n_planes: int = 16, n_tables: int = 1, seed: int = 42):
        self.n_planes = n_planes
        self.n_tables = n_tables
        self.rng = np.random.default_rng(seed)
        self.tables = []  

    def _signature(self, planes: np.ndarray, v: np.ndarray) -> tuple:
        proj = planes @ v
        return tuple(proj >= 0)

    def build(self, X: List[List[float]]):
        X_np = np.array(X, dtype=np.float32)
        _, d = X_np.shape
        self.tables = []
        for _ in range(self.n_tables):
            planes = self.rng.normal(size=(self.n_planes, d)).astype(np.float32)
            buckets = defaultdict(list)
            for i, vec in enumerate(X_np):
                sig = self._signature(planes, vec)
                buckets[sig].append(i)
            self.tables.append((planes, buckets))

    def get_candidates(self, q: List[float]) -> List[int]:
        q_np = np.array(q, dtype=np.float32)
        cands: Set[int] = set()
        for planes, buckets in self.tables:
            sig = self._signature(planes, q_np)
            cands.update(buckets.get(sig, []))
        return list(cands)
