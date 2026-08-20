import argparse
import glob
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

import cv2

from common import render_doc_page

IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
CLASSES = ["letter", "form", "email", "handwritten", "advertisement", "invoice", "memo", "resume"]


def _synthetic_page(cls_idx, seed):
    rng = random.Random(seed * 7 + cls_idx)
    img = render_doc_page(IMG_SIZE, seed=seed)
    img = np.asarray(img, dtype=np.uint8)
    bg = np.full_like(img, 255)
    bg[..., 0] = int((cls_idx * 23) % 60 + 180)
    bg[..., 1] = int((cls_idx * 31) % 50 + 190)
    bg[..., 2] = int((cls_idx * 17) % 45 + 200)
    img = cv2.addWeighted(bg, 0.55, img, 0.45, 0)
    cv2.rectangle(img, (0, 0), (IMG_SIZE, 26), ((cls_idx * 40) % 255, (cls_idx * 70 + 60) % 255, (cls_idx * 25 + 90) % 255), -1)
    for _ in range(3 + cls_idx % 4):
        x, y = rng.randint(10, IMG_SIZE - 40), rng.randint(10, IMG_SIZE - 40)
        w, h = rng.randint(20, 120), rng.randint(4, 14)
        cv2.rectangle(img, (x, y), (x + w, y + h), ((cls_idx * 20 + 30) % 255, (cls_idx * 15 + 40) % 255, (200 - cls_idx * 10) % 255), -1)
    return img


def synthetic_cls_items(per_class=200):
    items = []
    for c, cls in enumerate(CLASSES):
        for i in range(per_class):
            items.append((None, cls, c * per_class + i))
    return items


def build_model(num_classes):
    base = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(base.input, out)


def find_samples(data_dir, per_class=200):
    root = os.path.join(data_dir, "rvl_classes")
    items = []
    for c, cls in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(root, cls, "*.tif")))
        random.seed(42)
        random.shuffle(files)
        items.extend((f, cls, c * per_class + i) for i, f in enumerate(files[:per_class]))
    if not items:
        print("[cls] no RVL-CDIP data found -> using synthetic document pages")
        items = synthetic_cls_items(per_class=per_class)
    return items


def load_rgb(path, cls_idx, synth_seed=None):
    if path is None:
        return _synthetic_page(cls_idx, synth_seed) / 255.0
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.float32) / 255.0


def gen(items):
    for path, cls, seed in items:
        yield load_rgb(path, CLASSES.index(cls), seed), CLASSES.index(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    items = find_samples(args.data)
    if not items:
        raise SystemExit("no RVL-CDIP data found; run download_data.py first")
    print(f"[cls] {len(items)} images, {len(CLASSES)} classes")

    rng = random.Random(42)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * 0.15))
    val_items, train_items = items[:n_val], items[n_val:]

    train_ds = tf.data.Dataset.from_generator(
        lambda: gen(train_items),
        output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
            tf.TensorSpec((), tf.int64),
        ),
    ).map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        lambda: gen(val_items),
        output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
            tf.TensorSpec((), tf.int64),
        ),
    )
    train_ds = train_ds.batch(args.batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(AUTOTUNE)

    model = build_model(len(CLASSES))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    model.get_layer(model.layers[0].name).trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callbacks)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "cls_model.keras"))
    model.export(os.path.join(args.out, "cls_saved"))
    with open(os.path.join(args.out, "cls_labels.txt"), "w") as f:
        f.write("\n".join(CLASSES))
    print("[cls] done")


if __name__ == "__main__":
    main()