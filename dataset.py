"""
Datasets and transforms for MPA-FER.

Supports the two RAF-DB layouts people actually end up with:

  layout "official"  (the zip from the RAF-DB authors)
      <root>/basic/EmoLabel/list_patition_label.txt      "train_09748.jpg 5"
      <root>/basic/Image/aligned/train_09748_aligned.jpg

  layout "folder"    (most Kaggle mirrors)
      <root>/train/1/*.jpg ... <root>/train/7/*.jpg       (1..7 = RAF-DB label ids)
      <root>/test/1/*.jpg  ... <root>/test/7/*.jpg
      folder names may also be class names ("happiness", ...)

layout "auto" (default) sniffs the directory and picks one.

RAF-DB label ids are 1-based: 1 surprise, 2 fear, 3 disgust, 4 happiness,
5 sadness, 6 anger, 7 neutral. That is exactly the class_names order in
configs/rafdb.yaml, so internal index = raf_label - 1.
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.misc import seed_worker

# CLIP's own normalisation. Must match, or the frozen features are meaningless.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def build_transforms(cfg, train):
    """Train: RandomResizedCrop + flip + erasing (paper). Test: plain resize."""
    size = cfg["data"]["image_size"]
    normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)

    if not train:
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

    aug = cfg["data"]["augment"]
    ops = [
        transforms.RandomResizedCrop(
            size,
            scale=tuple(aug["random_resized_crop_scale"]),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
    ]
    if aug["horizontal_flip"]:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), normalize]
    if aug["random_erasing_p"] > 0:
        ops.append(transforms.RandomErasing(p=aug["random_erasing_p"]))
    return transforms.Compose(ops)


# ------------------------------------------------------------------
# Layout detection
# ------------------------------------------------------------------
def _label_file(root):
    for p in (
        os.path.join(root, "basic", "EmoLabel", "list_patition_label.txt"),
        os.path.join(root, "EmoLabel", "list_patition_label.txt"),
        os.path.join(root, "list_patition_label.txt"),
    ):
        if os.path.isfile(p):
            return p
    return None


def _image_dir(root):
    for p in (
        os.path.join(root, "basic", "Image", "aligned"),
        os.path.join(root, "Image", "aligned"),
        os.path.join(root, "aligned"),
        os.path.join(root, "basic", "Image", "original"),
    ):
        if os.path.isdir(p):
            return p
    return None


def detect_layout(root):
    if _label_file(root) and _image_dir(root):
        return "official"
    for split in ("train", "test", "Train", "Test", "DATASET/train"):
        if os.path.isdir(os.path.join(root, split)):
            return "folder"
    raise FileNotFoundError(
        f"Could not work out the dataset layout under '{root}'.\n"
        f"Contents: {sorted(os.listdir(root))[:20] if os.path.isdir(root) else 'path does not exist'}\n"
        "Expected either basic/EmoLabel/list_patition_label.txt, or train/ and test/ folders."
    )


def _find_split_dir(root, split):
    """split is 'train' or 'test'. Handles capitalisation and a DATASET/ wrapper."""
    candidates = [
        os.path.join(root, split),
        os.path.join(root, split.capitalize()),
        os.path.join(root, "DATASET", split),
        os.path.join(root, "DATASET", split.capitalize()),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"No '{split}' folder under '{root}' (tried {candidates})")


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class RAFDB(Dataset):
    """
    Returns (image_tensor, label_index). label_index is 0-based and follows
    cfg.data.class_names order.
    """

    def __init__(self, root, split, class_names, transform=None, layout="auto"):
        assert split in ("train", "test")
        self.root = root
        self.split = split
        self.class_names = list(class_names)
        self.transform = transform
        self.layout = detect_layout(root) if layout == "auto" else layout

        if self.layout == "official":
            self.samples = self._scan_official()
        elif self.layout == "folder":
            self.samples = self._scan_folder()
        else:
            raise ValueError(f"Unknown layout '{self.layout}'")

        if not self.samples:
            raise RuntimeError(f"Found 0 {split} images under '{root}' (layout={self.layout})")

    # ---------------- official layout ----------------
    def _scan_official(self):
        label_file = _label_file(self.root)
        img_dir = _image_dir(self.root)
        aligned = img_dir.endswith("aligned")

        samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, raf_label = line.split()
                if not name.startswith(self.split):  # "train_00001.jpg" / "test_0001.jpg"
                    continue
                if aligned:
                    stem, ext = os.path.splitext(name)
                    name = f"{stem}_aligned{ext}"
                path = os.path.join(img_dir, name)
                if os.path.isfile(path):
                    samples.append((path, int(raf_label) - 1))  # 1..7 -> 0..6
        return samples

    # ---------------- folder layout ----------------
    def _scan_folder(self):
        split_dir = _find_split_dir(self.root, self.split)
        samples = []
        for entry in sorted(os.listdir(split_dir)):
            sub = os.path.join(split_dir, entry)
            if not os.path.isdir(sub):
                continue
            label = self._folder_to_label(entry)
            for fn in sorted(os.listdir(sub)):
                if fn.lower().endswith(IMG_EXTS):
                    samples.append((os.path.join(sub, fn), label))
        return samples

    def _folder_to_label(self, name):
        key = name.strip().lower()
        if key.isdigit():  # "1".."7" are RAF-DB ids
            idx = int(key) - 1
            if not 0 <= idx < len(self.class_names):
                raise ValueError(f"Folder '{name}' is outside 1..{len(self.class_names)}")
            return idx
        # tolerate a few common spellings
        alias = {"happy": "happiness", "sad": "sadness", "angry": "anger",
                 "surprised": "surprise", "disgusted": "disgust", "fearful": "fear"}
        key = alias.get(key, key)
        if key not in self.class_names:
            raise ValueError(
                f"Folder '{name}' does not match any class in {self.class_names}. "
                "Rename it or fix data.class_names in the config."
            )
        return self.class_names.index(key)

    # ---------------- torch API ----------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        counts = [0] * len(self.class_names)
        for _, label in self.samples:
            counts[label] += 1
        return counts


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------