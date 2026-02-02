import time
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from embedding import load_embeddings
from search.KNN import BruteForceKNN
from search.LSH import RandomHyperplaneLSH


def benchmark_knn(X, queries, k=10):
    results = {}
    for metric in ["cosine", "euclidean"]:
        knn = BruteForceKNN(X, metric=metric)
        t0 = time.perf_counter()
        for q in queries:
            _idx, _score = knn.topk(q, k=k)
        t1 = time.perf_counter()
        results[metric] = (t1 - t0) / len(queries)
    return results


def benchmark_lsh(X, queries, k=10, n_planes=12, n_tables=2):
    lsh = RandomHyperplaneLSH(n_planes=n_planes, n_tables=n_tables)
    lsh.build(X)

    knn = BruteForceKNN(X, metric="cosine")

    total_time = 0.0
    recalls = []
    for q in queries:
        gt_idx, _ = knn.topk(q, k=k)
        t0 = time.perf_counter()
        candidates = lsh.get_candidates(q)
        top = None
        if candidates:
            C = X[candidates]
            sims = C @ q
            order = np.argsort(-sims)[:k]
            top = [candidates[i] for i in order]
        else:
            top = []
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        recall = len(set(gt_idx).intersection(top)) / k
        recalls.append(recall)

    avg_time = total_time / len(queries)
    avg_recall = float(np.mean(recalls))
    return avg_time, avg_recall, len(set().union(*[set(lsh.get_candidates(q)) for q in queries]))

def write_excel_csv(rows, out_path: Path):
    header = ["method", "metric", "k", "n_planes", "n_tables", "avg_time_sec", "avg_recall"]
    lines = [",".join(header)]
    for row in rows:
        line = ",".join(str(row.get(col, "")) for col in header)
        lines.append(line)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    X, _ = load_embeddings()
    rng = np.random.default_rng(0)
    num_queries = 20
    k = 10
    idx = rng.integers(0, X.shape[0], size=num_queries)
    queries = X[idx]

    knn_times = benchmark_knn(X, queries, k=k)
    lsh_time, lsh_recall, _ = benchmark_lsh(X, queries, k=k, n_planes=12, n_tables=2)

    rows = []
    for metric, t in knn_times.items():
        rows.append({
            "method": "knn",
            "metric": metric,
            "k": k,
            "n_planes": "",
            "n_tables": "",
            "avg_time_sec": f"{t:.6f}",
            "avg_recall": "",
        })
    rows.append({
        "method": "lsh",
        "metric": "cosine_rerank",
        "k": k,
        "n_planes": 12,
        "n_tables": 2,
        "avg_time_sec": f"{lsh_time:.6f}",
        "avg_recall": f"{lsh_recall:.2f}",
    })
    out_csv = Path("tests/benchmark_results.csv")
    write_excel_csv(rows, out_csv)

    print("k-NN avg query time (s):")
    for metric, t in knn_times.items():
        print(f"  {metric}: {t:.6f}")
    print(f"LSH avg query time (s): {lsh_time:.6f} (n_planes=12, n_tables=2, recall@{k}={lsh_recall:.2f})")
    print("Big-O (theoretical):")
    print("  Brute-force kNN: O(N*d) per query")
    print("  LSH: O(n_planes*d*n_tables) to hash + candidates*r for rerank (sub-linear expected in practice)")
    print(f"Wrote CSV for Excel: {out_csv}")


if __name__ == "__main__":
    main()
