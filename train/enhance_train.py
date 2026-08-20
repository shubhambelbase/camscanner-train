import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import degrade, render_doc_page

IMG_SIZE = 256
AUTOTUNE = tf.data.AUTOTUNE


def build_model():
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    c1 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    c1 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(c1)
    p1 = keras.layers.MaxPool2D()(c1)
    c2 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    c2 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(c2)
    p2 = keras.layers.MaxPool2D()(c2)
    c3 = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    c3 = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(c3)
    p3 = keras.layers.MaxPool2D()(c3)
    c4 = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(p3)
    c4 = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(c4)

    u = keras.layers.UpSampling2D(2)(c4)
    u = keras.layers.Concatenate()([u, c3])
    u = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(u)
    u = keras.layers.UpSampling2D(2)(u)
    u = keras.layers.Concatenate()([u, c2])
    u = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(u)
    u = keras.layers.UpSampling2D(2)(u)
    u = keras.layers.Concatenate()([u, c1])
    u = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(u)
    out = keras.layers.Conv2D(3, 1, activation="sigmoid")(u)
    return keras.Model(inputs, out)


def gen_pairs(batch_size, seed):
    rng = np.random.default_rng(seed)
    while True:
        batch_in = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        batch_out = np.zeros_like(batch_in)
        for i in range(batch_size):
            clean = render_doc_page(IMG_SIZE, seed=int(rng.integers(0, 2**31)))
            degraded = degrade(clean, seed=int(rng.integers(0, 2**31)))
            batch_in[i] = degraded.astype(np.float32) / 255.0
            batch_out[i] = clean.astype(np.float32) / 255.0
        yield batch_in, batch_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    args = ap.parse_args()

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(2e-4),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    train_gen = gen_pairs(args.batch_size, seed=1)
    val_gen = gen_pairs(args.batch_size, seed=2)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        validation_steps=20,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "enhance_model.keras"))
    model.export(os.path.join(args.out, "enhance_saved"))
    print("[enhance] done")


if __name__ == "__main__":
    main()