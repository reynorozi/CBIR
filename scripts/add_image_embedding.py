import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


def load_model(device: torch.device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def embed_image(img_path: Path, model, preprocess, device: torch.device) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(t).squeeze().cpu().numpy()
    return vec.astype(np.float32)


def normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return (v / norm).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Append a new image embedding to .npy files.")
    parser.add_argument("--image", required=True, help="Path to the image to embed.")
    parser.add_argument("--base-dir", default=".", help="Base dir to make the stored image path relative.")
    parser.add_argument("--output-dir", default="data", help="Directory containing the embedding .npy files.")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "caltech101_embeddings.npy"
    norm_path = out_dir / "caltech101_embeddings_norm.npy"
    ids_path = out_dir / "caltech101_image_ids.npy"

    raw = np.load(raw_path).astype(np.float32) if raw_path.exists() else None
    norm = np.load(norm_path).astype(np.float32) if norm_path.exists() else None
    ids = np.load(ids_path, allow_pickle=True).tolist() if ids_path.exists() else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_model(device)

    vec = embed_image(img_path, model, preprocess, device)
    norm_vec = normalize_vec(vec)

    if raw is None:
        raw = vec.reshape(1, -1)
    else:
        raw = np.vstack([raw, vec])
    np.save(raw_path, raw)

    if norm is None:
        norm = norm_vec.reshape(1, -1)
    else:
        norm = np.vstack([norm, norm_vec])
    np.save(norm_path, norm)

    try:
        rel_path = img_path.relative_to(base_dir)
    except ValueError:
        rel_path = img_path
    rel_str = str(rel_path).replace("\\", "/")
    ids.append(rel_str)
    np.save(ids_path, np.array(ids, dtype=object))

    print("Added image")
    print("image:", img_path)
    print("stored path:", rel_str)
    print("raw embeddings:", raw.shape, "norm embeddings:", norm.shape)


if __name__ == "__main__":
    main()
