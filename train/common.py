import random

import cv2
import numpy as np


def render_doc_page(size=256, seed=None):
    rng = random.Random(seed)
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    y = 30
    while y < size - 40:
        w = int(size * rng.uniform(0.55, 0.9))
        x0 = int(size * rng.uniform(0.02, 0.12))
        h = max(3, int(size * rng.uniform(0.008, 0.02)))
        cv2.rectangle(img, (x0, y), (x0 + w, y + h), (170, 170, 175), -1)
        y += int(h * rng.uniform(2.2, 3.4))
    return img


def random_quad(size, margin=0.08, seed=None):
    rng = random.Random(seed)
    m = int(size * margin)
    lo, hi = m, size - m
    pts = np.array(
        [
            [rng.randint(lo, int(size * 0.55)), rng.randint(lo, int(size * 0.55))],
            [rng.randint(int(size * 0.45), hi), rng.randint(lo, int(size * 0.55))],
            [rng.randint(int(size * 0.45), hi), rng.randint(int(size * 0.45), hi)],
            [rng.randint(lo, int(size * 0.55)), rng.randint(int(size * 0.45), hi)],
        ],
        dtype=np.float32,
    )
    return pts


def synthetic_doc_pair(size=256, seed=None):
    rng = random.Random(seed)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(40):
        c = rng.randint(40, 110)
        x, y = rng.randint(0, size), rng.randint(0, size)
        r = rng.randint(6, 60)
        cv2.circle(bg, (x, y), r, (c, c, c), -1)
    quad = random_quad(size, seed=rng.random())
    page = render_doc_page(size, seed=rng.random())
    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    h = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(page, h, (size, size))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out, mask, warped


def degrade(img, seed=None):
    rng = random.Random(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    shadow = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        rx, ry = rng.uniform(0.25, 0.7) * w, rng.uniform(0.25, 0.7) * h
        angle = rng.uniform(0, 180)
        a = np.exp(-(((np.arange(w)[None, :] - cx) * np.cos(angle) + (np.arange(h)[:, None] - cy) * np.sin(angle)) ** 2) / (2 * rx ** 2))
        b = np.exp(-((-(np.arange(w)[None, :] - cx) * np.sin(angle) + (np.arange(h)[:, None] - cy) * np.cos(angle)) ** 2) / (2 * ry ** 2))
        shadow += rng.uniform(0.35, 0.8) * (a * b) ** 2
    shadow = np.clip(shadow, 0, 0.85)
    out = out * (1.0 - shadow[..., None])
    out = out * rng.uniform(0.75, 1.05) + rng.uniform(-8, 8)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.5, 1.5))
    return out


def apply_geometric_aug(img, mask=None, seed=None):
    rng = random.Random(seed)
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
    if mask is not None:
        pts = random_quad(h * 2, margin=0.15, seed=rng.random()) * 0.5
        m = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32))
        img = cv2.warpPerspective(img, m, (w, h))
        mask = cv2.warpPerspective(mask, m, (w, h))
    return img, mask


def split_train_val(items, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = max(1, int(len(items) * val_frac))
    return items[n:], items[:n]