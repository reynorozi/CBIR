# CRIB

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
