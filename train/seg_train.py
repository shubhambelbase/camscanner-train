import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import apply_geometric_aug, split_train_val, synthetic_doc_pair

IMG_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = True
    x = base.get_layer("block_16_add").output
    for filters in (256, 128, 64, 32):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D(2)(x)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    out = keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(base.input, out)


def find_midv_samples(data_dir, max_samples=3000):
    quads = sorted(glob.glob(os.path.join(data_dir, "midv500", "**", "*_quad.json"), recursive=True))
    items = []
    for qp in quads:
        img = qp.replace("_quad.json", ".jpg")
        if not os.path.exists(img):
            continue
        with open(qp) as f:
            quad = json.load(f)["quad"]
        items.append((img, np.asarray(quad, dtype=np.float32), None))
    random.seed(42)
    random.shuffle(items)
    return items[:max_samples]


def make_mask(quad, h, w):
    pts = quad * np.array([w / IMG_SIZE, h / IMG_SIZE], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def synthetic_items(n=1500):
    items = []
    for i in range(n):
        items.append((None, None, i))
    return items


def load_synthetic(idx):
    img, mask, _ = synthetic_doc_pair(IMG_SIZE, seed=idx)
    mask = (mask > 0).astype(np.uint8)
    return img, mask


def load_pair(item):
    img_path, quad, idx = item
    if img_path is None:
        return load_synthetic(idx)
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = make_mask(quad, IMG_SIZE, IMG_SIZE)
    return img, mask


def gen(items):
    for item in items:
        pair = load_pair(item)
        if pair is None:
            continue
        img, mask = pair
        yield img.astype(np.float32) / 255.0, mask[..., None].astype(np.float32)


def augment(img, mask):
    img, mask = tf.py_function(
        lambda i, m: apply_geometric_aug(i.numpy(), m.numpy()),
        [img, mask],
        Tout=[tf.float32, tf.float32],
    )
    img = tf.image.random_brightness(img, 0.2)
    return img, mask


def make_dataset(items, batch_size, augment_on):
    ds = tf.data.Dataset.from_generator(
        lambda: gen(items), output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 1), tf.float32),
        )
    )
    if augment_on:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    items = find_midv_samples(args.data)
    if not items:
        print("[seg] no MIDV-500 data found -> using synthetic document pairs")
        items = synthetic_items()
    print(f"[seg] {len(items)} training frames")

    train_items, val_items = split_train_val(items, val_frac=0.1)
    train_ds = make_dataset(train_items, args.batch_size, augment_on=True)
    val_ds = make_dataset(val_items, args.batch_size, augment_on=False)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryIoU(target_class_ids=[0])],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "seg_model.keras"))
    model.export(os.path.join(args.out, "seg_saved"))
    print("[seg] done")


if __name__ == "__main__":
    main()