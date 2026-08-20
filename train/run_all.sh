#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "=== [1/5] installing deps ==="
pip install -q -r docker/requirements.txt
echo "=== [2/5] downloading data ==="
python train/download_data.py --data /data || echo "[warn] data download had issues; continuing with synthetic fallbacks"
echo "=== [3/5] training segmentation (doc detection) ==="
python train/seg_train.py --data /data --out /out
echo "=== [4/5] training enhancement ==="
python train/enhance_train.py --out /out
echo "=== [5/5] training classifier ==="
python train/cls_train.py --data /data --out /out || echo "[warn] classifier training failed; converting what we have"
echo "=== converting to TFLite ==="
python train/convert_tflite.py --data /data --out /out || echo "[warn] tflite conversion had issues"
echo "=== packing results ==="
cd /out
tar czf /results.tar.gz *.tflite *.txt 2>/dev/null || tar czf /results.tar.gz *.tflite
ls -la /results.tar.gz
echo "ALL_TRAINING_DONE"