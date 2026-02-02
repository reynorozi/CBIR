import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18, ResNet18_Weights
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from core.VectorEntity import VectorEntity
from infrastructure.VectorDB import VectorDB
from search.KNN import BruteForceKNN
from search.LSH import RandomHyperplaneLSH

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
DB_FILE = DATA_DIR / "vector_store.sqlite"

def normalize_path(p: str) -> str:
    return p.replace("\\", "/")

def init_db_if_needed():
    if not DB_FILE.exists():
        vectors_file = DATA_DIR / "caltech101_embeddings_norm.npy"
        ids_file = DATA_DIR / "caltech101_image_ids.npy"
        vectors = np.load(vectors_file).astype(np.float32)
        image_ids = np.load(ids_file, allow_pickle=True).tolist()
        db = VectorDB(DB_FILE)
        for i, vec in enumerate(vectors):
            img_path = normalize_path(str(image_ids[i]))
            entity = VectorEntity(vector=vec.tolist(), image_path=img_path)
            db.add_entity(entity)
    return VectorDB(DB_FILE)

db = init_db_if_needed()

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, device

model, preprocess, device = load_model()

def embed_image(img: Image.Image) -> list:
    t = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(t).squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-12
    return (vec / norm).tolist()

def cosine_rerank(X, q, candidates, k):
    if len(candidates) == 0:
        return []
    C = np.array([X[i] for i in candidates], dtype=np.float32)
    q_vec = np.array(q, dtype=np.float32)
    sims = C @ q_vec
    top = np.argsort(-sims)[:k]
    return [candidates[i] for i in top]

@st.cache_data(show_spinner=False)
def pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T

st.set_page_config(page_title="Caltech101 Vector DB Search", layout="wide")
st.title("Caltech101 Image Similarity Search with CRUD")

st.sidebar.header("Upload & Add New Image")
uploaded_file = st.sidebar.file_uploader("Upload image", type=["jpg","png","jpeg"])
label_input = st.sidebar.text_input("Labels (comma separated)")

if st.sidebar.button("Add Image"):
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        vec = embed_image(img)
        labels = [l.strip() for l in label_input.split(",")] if label_input else []
        img_path = Path("uploaded_images") / uploaded_file.name
        img_path.parent.mkdir(exist_ok=True, parents=True)
        img.save(img_path)
        entity = VectorEntity(vector=vec, labels=labels, image_path=str(img_path))
        db.add_entity(entity)
        st.sidebar.success(f"Image added with ID: {entity.id}")
    else:
        st.sidebar.warning("Upload an image first.")

st.sidebar.header("Edit/Delete Uploaded Images")
uploaded_entities = [e for e in db.entities if str(e.image_path).startswith("uploaded_images")]

if uploaded_entities:
    selected_id = st.sidebar.selectbox(
        "Select your uploaded image to Edit/Delete",
        [e.id for e in uploaded_entities],
        format_func=lambda eid: Path(db.get_entity(eid).image_path).name
    )
    if st.sidebar.button("Delete Image"):
        db.delete_entity(selected_id)
        st.sidebar.success("Image deleted")

    new_labels = st.sidebar.text_input("Edit Labels (comma separated)")
    if st.sidebar.button("Update Labels"):
        labels_list = [l.strip() for l in new_labels.split(",")] if new_labels else []
        db.update_entity(selected_id, new_labels=labels_list)
        st.sidebar.success("Labels updated")
else:
    st.sidebar.info("No images uploaded by you yet.")

mode = st.radio("Search mode", ["Brute-force KNN (exact)", "LSH + cosine rerank (approx)"])
k = st.slider("k", 1, 20, 10)
query_mode = st.radio("Query source", ["Dataset index", "Upload image"])

col1, col2 = st.columns([1, 2])
X = db.vectors
image_ids = db.ids

with col1:
    st.subheader("Query")
    upload_query = None
    idx = 0
    if query_mode == "Dataset index":
        if len(image_ids) == 0:
            st.warning("Database is empty.")
        else:
            idx = st.number_input("Query index", min_value=0, max_value=len(image_ids)-1, value=0)
            q_path = Path(normalize_path(db.image_paths[idx]))
            if q_path.exists():
                st.image(str(q_path), caption=f"Query: {image_ids[idx]}", use_container_width=True)
    else:
        upload_query = st.file_uploader("Upload query image", type=["jpg","jpeg","png"])
        if upload_query:
            img = Image.open(upload_query)
            st.image(img, caption="Uploaded query", use_container_width=True)

with col2:
    st.subheader("Results")
    if st.button("Search"):
        if query_mode == "Dataset index":
            q_vec = X[idx]
        else:
            if upload_query is None:
                st.warning("Upload an image first.")
                st.stop()
            img = Image.open(upload_query)
            q_vec = embed_image(img)

        if mode.startswith("Brute"):
            knn = BruteForceKNN(X)
            top_idx, sims = knn.topk(q_vec, k=k)
            sims = sims.tolist()
        else:
            lsh = RandomHyperplaneLSH(n_planes=12)
            if len(X) > 0:
                lsh.build(X)
                candidates = lsh.get_candidates(q_vec)
                top_idx = cosine_rerank(X, q_vec, candidates, k)
            else:
                top_idx = []

        cols = st.columns(5)
        for i, ridx in enumerate(top_idx):
            imgp = Path(normalize_path(db.image_paths[ridx]))
            caption = f"rank {i+1}"
            if mode.startswith("Brute") and sims:
                caption += f" | score {sims[i]:.3f}"
            if imgp.exists():
                with cols[i % 5]:
                    st.image(str(imgp), caption=caption, use_container_width=True)
            else:
                st.caption(f"Missing: {imgp}")

st.divider()
st.subheader("Embedding Scatter (PCA 2D)")
show_scatter = st.checkbox("Show scatter plot", value=False)
if show_scatter:
    max_points = st.slider("Max points", min_value=200, max_value=9000, value=1500, step=100)
    X_np = np.array(db.vectors, dtype=np.float32)
    if X_np.size == 0:
        st.info("Database is empty.")
    else:
        if X_np.shape[0] > max_points:
            idxs = np.random.default_rng(0).choice(X_np.shape[0], size=max_points, replace=False)
            X_plot = X_np[idxs]
        else:
            X_plot = X_np
        coords = pca_2d(X_plot)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.6)
        ax.set_title("PCA projection of embeddings")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
