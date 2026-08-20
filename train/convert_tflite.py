import argparse
import glob
import os

import numpy as np
import tensorflow as tf

from common import degrade, render_doc_page

IMG_SIZE = 256


def rep_seg(data_dir):
    quads = sorted(glob.glob(os.path.join(data_dir, "midv500", "**", "*_quad.json"), recursive=True))
    import cv2
    from seg_train import IMG_SIZE as S
    count = 0
    for qp in quads[:40]:
        img = qp.replace("_quad.json", ".jpg")
        if not os.path.exists(img):
            continue
        im = cv2.imread(img)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (S, S)).astype(np.float32) / 255.0
        yield [im]
        count += 1
        if count >= 20:
            break


def rep_enhance():
    for i in range(20):
        clean = render_doc_page(IMG_SIZE, seed=i)
        degraded = degrade(clean, seed=i)
        yield [degraded.astype(np.float32) / 255.0]


def rep_cls(data_dir):
    import cv2
    from cls_train import CLASSES, IMG_SIZE as S, _synthetic_page
    root = os.path.join(data_dir, "rvl_classes")
    count = 0
    for c, cls in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(root, cls, "*.tif")))[:4]:
            im = cv2.imread(f)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (S, S)).astype(np.float32) / 255.0
            yield [im]
            count += 1
        if count >= 20:
            break
    if count == 0:
        for i in range(20):
            yield [_synthetic_page(i % len(CLASSES), i).astype(np.float32) / 255.0]


def convert(name, saved_dir, out_dir, rep_fn):
    print(f"[tflite] converting {name}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_fn()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    path = os.path.join(out_dir, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite)
    print(f"[tflite] {name} -> {path} ({len(tflite)/1024/1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert("seg", os.path.join(args.out, "seg_saved"), args.out, lambda: rep_seg(args.data))
    convert("enhance", os.path.join(args.out, "enhance_saved"), args.out, rep_enhance)
    convert("cls", os.path.join(args.out, "cls_saved"), args.out, lambda: rep_cls(args.data))
    print("[tflite] all done")


if __name__ == "__main__":
    main()