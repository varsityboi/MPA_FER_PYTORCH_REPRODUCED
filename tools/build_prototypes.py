"""
Build the class prototypes of paper Eq. 6.

    p_c = mean over training images of class c of  z^g_i

where z^g is the CLS feature of the FROZEN CLIP image encoder (no visual
prompts, no augmentation). Nothing here is learned, so this runs once and
gets cached; train.py calls it automatically if the file is missing.

    python tools/build_prototypes.py
    python tools/build_prototypes.py --set losses.proto_align.samples_per_class=16
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import build_dataset
from models.mpa_fer import build_model
from utils.misc import get_device, load_config, set_seed


def prototypes_path(cfg):
    """Where the cache lives. Config can override with losses.proto_align.prototypes_file."""
    explicit = cfg["losses"]["proto_align"].get("prototypes_file")
    if explicit:
        return explicit
    n = cfg["losses"]["proto_align"]["samples_per_class"]
    tag = "full" if n is None or n < 0 else f"n{n}"
    backbone = cfg["model"]["clip_backbone"].replace("/", "").replace("-", "").lower()
    return os.path.join(cfg["output"]["dir"], f"prototypes_{backbone}_{tag}.pt")


def subset_indices(dataset, samples_per_class, n_classes, seed=42):
    """Pick at most N images per class (Table 3 ablation). -1 or None = keep everything."""
    if samples_per_class is None or samples_per_class < 0:
        return list(range(len(dataset)))
    g = torch.Generator().manual_seed(seed)
    per_class = [[] for _ in range(n_classes)]
    for i, (_, label) in enumerate(dataset.samples):
        per_class[label].append(i)
    chosen = []
    for idxs in per_class:
        idxs = torch.tensor(idxs)
        perm = torch.randperm(len(idxs), generator=g)[:samples_per_class]
        chosen += idxs[perm].tolist()
    return sorted(chosen)


@torch.no_grad()
def compute_prototypes(model, cfg, device, samples_per_class=None, verbose=True):
    """Returns a (C, embed_dim) float32 tensor on CPU."""
    class_names = cfg["data"]["class_names"]
    n_cls = len(class_names)
    if samples_per_class is None:
        samples_per_class = cfg["losses"]["proto_align"]["samples_per_class"]

    # No augmentation: prototypes should describe the class, not the crop.
    dataset = build_dataset(cfg, "train", train_transform=False)
    idxs = subset_indices(dataset, samples_per_class, n_cls, seed=cfg["seed"])
    loader = DataLoader(
        Subset(dataset, idxs),
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model.eval()
    dim = model.text_prompts.hard_text_features.shape[1]
    sums = torch.zeros(n_cls, dim, dtype=torch.float64, device=device)
    counts = torch.zeros(n_cls, dtype=torch.float64, device=device)

    it = tqdm(loader, desc="prototypes", disable=not verbose)
    for images, labels in it:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            feats = model.encode_frozen_image(images)
        feats = feats.double()
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))

    if (counts == 0).any():
        missing = [class_names[i] for i in (counts == 0).nonzero().flatten().tolist()]
        raise RuntimeError(f"No training images found for class(es): {missing}")

    protos = (sums / counts.unsqueeze(1)).float().cpu()
    if verbose:
        print(f"  used {int(counts.sum().item())} images; per class {counts.int().tolist()}")
        print(f"  prototype norms: {[round(v, 2) for v in protos.norm(dim=-1).tolist()]}")
    return protos


def load_or_build(cfg, model, device, logger=None, force=False):
    """Used by train.py. Loads the cache if it exists, otherwise builds and saves it."""
    path = prototypes_path(cfg)
    say = logger.log if logger else print
    n_cls = len(cfg["data"]["class_names"])

    if os.path.isfile(path) and not force:
        protos = torch.load(path, map_location="cpu", weights_only=True)
        if protos.shape[0] != n_cls:
            raise ValueError(f"Cached prototypes at {path} have {protos.shape[0]} classes, expected {n_cls}")
        say(f"Loaded prototypes from {path}")
        return protos

    say("Building class prototypes (Eq. 6) with the frozen CLIP encoder...")
    protos = compute_prototypes(model, cfg, device)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(protos, path)
    say(f"Saved prototypes to {path}")
    return protos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rafdb.yaml")
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    p.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = p.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    device = get_device()
    model = build_model(cfg, device)
    load_or_build(cfg, model, device, force=args.force)


if __name__ == "__main__":
    main()