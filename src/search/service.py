from functools import lru_cache
from typing import List, Tuple
import numpy as np
from infrastructure.VectorDB import VectorDB
from core.VectorEntity import VectorEntity
from search.KNN import BruteForceKNN
from search.LSH import RandomHyperplaneLSH

DB_FILE = "data/vector_store.sqlite"

@lru_cache(maxsize=1)
def get_db() -> VectorDB:
    return VectorDB(DB_FILE)

@lru_cache(maxsize=8)
def get_knn(metric="cosine") -> BruteForceKNN:
    db = get_db()
    return BruteForceKNN(db.vectors, metric)

@lru_cache(maxsize=8)
def get_lsh(n_planes=12, n_tables=1) -> RandomHyperplaneLSH:
    db = get_db()
    lsh = RandomHyperplaneLSH(n_planes, n_tables)
    lsh.build(db.vectors)
    return lsh

def search_knn_by_index(query_index: int, k=10, metric="cosine") -> List[Tuple[int, float, str]]:
    db = get_db()
    knn = get_knn(metric)
    q_vec = db.vectors[query_index]
    idxs, scores = knn.topk(q_vec, k)
    return [(int(i), float(scores[j]), db.ids[i]) for j, i in enumerate(idxs)]

def search_lsh_by_index(query_index: int, k=10, n_planes=12, n_tables=1) -> List[Tuple[int, str]]:
    db = get_db()
    q_vec = db.vectors[query_index]
    lsh = get_lsh(n_planes, n_tables)
    candidates = lsh.get_candidates(q_vec)
    if not candidates:
        return []
    C = np.array([db.vectors[i] for i in candidates], dtype=np.float32)
    sims = C @ np.array(q_vec, dtype=np.float32)
    top = np.argsort(-sims)[:k]
    return [(candidates[i], db.ids[candidates[i]]) for i in top]
