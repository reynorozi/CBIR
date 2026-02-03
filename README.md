# CBIR

## How to Run

1) Create/activate a venv (optional):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2) Install deps:
   - `pip install streamlit numpy pillow matplotlib`
   - PyTorch (CPU): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
3) Run the app:
   - `streamlit run streamlit_app.py`

The first run will initialize the SQLite database from the precomputed embeddings in `data/`.


## Vector Storage and Persistence

- Persistent store lives at `data/vector_store.sqlite` (created on first write).
- Table `vectors` holds:
  - `id`: UUID string per vector (primary key).
  - `vector`: float32 embedding stored as a BLOB.
  - `labels`: JSON list of labels.
  - `image_path`: relative/absolute path to the image.
- Format: SQLite on disk. In-memory lists are used as a working buffer; the on-disk DB is authoritative.

### How vectors/metadata are stored and loaded
- `src/infrastructure/VectorDB.py` loads the SQLite DB on init and rehydrates `VectorEntity` objects from the `vectors` table.
- Writes go through `INSERT/UPDATE/DELETE` on the SQLite table.
- Metadata association: each row stores its own `id`, `labels`, and `image_path` alongside the vector.

### CRUD Operations
- Create: `add_entity(entity)` appends vector + metadata, assigns/keeps UUID, and persists.
- Read: `get_by_id(id)` returns the matching `VectorEntity`; `get_all_entities()` returns all.
- Update: `update(id, new_entity)` selectively overwrites non-None fields (vector/labels/image_path) and persists.
- Delete: `delete(id)` removes by id and persists.

### Precomputed Embeddings
- Read-only embeddings live in `data/caltech101_embeddings_norm.npy` with ids in `data/caltech101_image_ids.npy`.
- Use `embedding.load_embeddings()` to load those for search; they are independent from the CRUD store above.

## Project Layout
- `src/` contains the application modules (`core`, `infrastructure`, `search`, `embedding`).

## Scripts
- Add a new embedding to the precomputed files:
  - `python3 scripts/add_image_embedding.py --image /path/to/img.jpg --base-dir . --output-dir data`
- Normalize raw embeddings:
  - `python3 scripts/normalize_embeddings.py`
 
## Reports

- Project report: `reports/CBIR_LSH_Report.docx`
- Assignment PDF: `reports/Content_Based_Image_Retrieval_Using_Locality_Sensitive_Hashing.pdf`

## GUI Features

- Upload an image to add it to the SQLite DB.
- Search by dataset index or by uploading a query image.
- Exact k-NN (cosine/euclidean) or approximate LSH search.
- PCA 2D scatter plot of embeddings (toggle in the UI).

## Benchmarking

Run:

- `python3 tests/benchmark_search.py`

This writes an Excel-friendly CSV at `tests/benchmark_results.csv` with k-NN and LSH timings/recall.

## k-NN and LSH Benchmarks (Caltech101 embeddings)

Ran `python3 tests/benchmark_search.py` (20 random queries, k=10):

- kNN cosine avg query time: ~0.0004 s
- kNN Euclidean avg query time: ~0.0094 s
- LSH (12 planes, 2 tables) avg query time: ~0.00016 s, recall@10 ≈ 0.29 vs. cosine ground truth
- Theory: brute-force kNN is O(N·d) per query; LSH is O(n_planes·d·n_tables) for hashing plus candidate rerank (expected sub-linear).

Note: LSH recall can be improved by tuning `n_planes`/`n_tables` at some cost to speed; current settings favor speed over accuracy.

