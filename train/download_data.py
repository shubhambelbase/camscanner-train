import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

RVL_ARCHIVE = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz"
RVL_TRAIN_TXT = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data/train.txt"
RVL_CLASSES = ["letter", "form", "email", "handwritten", "advertisement", "invoice", "memo", "resume"]


def download_midv(data_dir):
    target = os.path.join(data_dir, "midv500")
    if os.path.isdir(target) and any(os.scandir(target)):
        print(f"[midv] found existing dataset at {target}")
        return target
    try:
        import midv500
    except ImportError:
        print("[midv] 'midv500' package missing -> installing")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "midv500"])
        import midv500
    print("[midv] downloading MIDV-500 via midv500 package (~7GB, slow on first run)")
    midv500.download_dataset(data_dir, "midv500")
    print(f"[midv] done -> {target}")


def download_rvl(data_dir, per_class=200):
    classes = {i: RVL_CLASSES[i] for i in range(len(RVL_CLASSES))}
    class_names = list(classes.values())
    label_path = os.path.join(data_dir, "rvl-train.txt")
    if not os.path.exists(label_path):
        print("[rvl] downloading train.txt label file")
        urllib.request.urlretrieve(RVL_TRAIN_TXT, label_path)
    selected = {c: 0 for c in class_names}
    members = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            path, label = parts[0], int(parts[1])
            cls = classes.get(label)
            if cls and selected[cls] < per_class:
                selected[cls] += 1
                members.append((path, cls))
    if not members:
        print("[rvl] no members matched; check label file format")
        sys.exit(1)
    print(f"[rvl] selected {len(members)} files: {selected}")
    archive = os.path.join(data_dir, "rvl-cdip.tar.gz")
    if not os.path.exists(archive):
        print(f"[rvl] downloading archive (~38GB) from HuggingFace")
        urllib.request.urlretrieve(RVL_ARCHIVE, archive)
    out_dir = os.path.join(data_dir, "rvl")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    print("[rvl] extracting only selected members from tarball")
    with tarfile.open(archive, "r:gz") as tf:
        names = {m[0] for m in members}
        for m in tf.getmembers():
            if m.name in names:
                m.name = os.path.basename(m.name)
                tf.extract(m, path=out_dir)
    print("[rvl] done")


def build_rvl_class_dirs(data_dir, per_class=200):
    classes = {i: RVL_CLASSES[i] for i in range(len(RVL_CLASSES))}
    label_path = os.path.join(data_dir, "rvl-train.txt")
    if not os.path.exists(label_path):
        return None
    selected = {c: 0 for c in classes.values()}
    mapping = {}
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            path, label = parts[0], int(parts[1])
            cls = classes.get(label)
            if cls and selected[cls] < per_class:
                selected[cls] += 1
                mapping[path] = cls
    out_root = os.path.join(data_dir, "rvl_classes")
    for cls in classes.values():
        os.makedirs(os.path.join(out_root, cls), exist_ok=True)
    return mapping, out_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--per-class", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    download_midv(args.data)
    download_rvl(args.data, per_class=args.per_class)
    build_rvl_class_dirs(args.data, per_class=args.per_class)
    print("[ok] all data ready")


if __name__ == "__main__":
    main()