import numpy as np

INPUT_PATH = "data/caltech101_embeddings.npy"
OUTPUT_PATH = "data/caltech101_embeddings_norm.npy"

X = np.load(INPUT_PATH).astype(np.float32)

# normalize each vector to length 1
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

np.save(OUTPUT_PATH, X)

print("✅ saved:", OUTPUT_PATH)
print("shape:", X.shape, "dtype:", X.dtype)
print("first 5 lengths:", np.linalg.norm(X[:5], axis=1))