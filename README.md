# CamScanner Mobile Models — Fine-tuning pipeline

Three TFLite models for a cam-scanner Android app:

| Model | File | Task |
|---|---|---|
| A. Segmentation | `seg.tflite` | Document boundary detection (mask → corners in app) |
| B. Enhancement | `enhance.tflite` | Shadow removal / contrast (UNet, trained on synthetic pairs) |
| C. Classification | `cls.tflite` | Document type (letter/form/email/handwritten/ad/invoice/memo/resume) |

All three are fine-tuned on top of pretrained backbones (MobileNetV2 / MobileNetV3),
then INT8-quantized to run fast on mobile.

## Datasets

- **MIDV-500** (~7GB, 15k frames of ID documents with corner annotations) — model A
  Downloaded automatically by `download_data.py` via the `midv500` pip package.
  If the FTP source is slow/unavailable, a fallback mirror URL can be dropped in later.
- **RVL-CDIP** (38GB tarball) — model C. The script downloads the *label file* only and
  extracts just the ~1600 selected images from the tarball, so disk usage stays small.
- **Enhancement** — no download: training pairs (clean vs degraded-with-shadows) are
  generated synthetically on the GPU, so this step costs ~0 data bandwidth.

## Layout

```
train/
  download_data.py   # fetch MIDV-500 + RVL-CDIP subset
  seg_train.py       # model A
  enhance_train.py   # model B
  cls_train.py       # model C
  convert_tflite.py  # -> 3x .tflite (INT8)
  run_all.sh         # 1 command: data -> train all -> convert
  common.py          # shared aug + synthetic page renderer
docker/
  Dockerfile         # tensorflow/tensorflow:2.16.1-gpu
  requirements.txt
```

## Run on Vast.ai (the $4 plan)

1. **Find a GPU** (re-query prices first — they fluctuate):
   - Filter: `RTX 4070S` or `RTX 3060`, on-demand, 1 GPU, >=8GB VRAM, >=60GB disk
   - Budget target: `<= $0.10/hr`
2. **Launch** an instance with the PyTorch/TensorFlow docker template (CUDA image),
   or build from `docker/Dockerfile`.
3. **Upload this folder**:
   ```
   scp -r camscanner root@<instance_ip>:/workspace/
   ```
4. **Run everything** (detached so SSH drops don't waste money):
   ```
   nohup bash /workspace/camscanner/train/run_all.sh > /workspace/train.log 2>&1 &
   ```
5. **Watch progress**:
   ```
   tail -f /workspace/train.log
   ```
6. When you see `ALL_TRAINING_DONE`, download results:
   ```
   scp -r root@<instance_ip>:/out ./models
   ```
7. **Destroy the instance immediately** — every hour idle burns credit.

## Budget estimate

- Total GPU time: ~4–6 hrs on RTX 4070S-class GPU (all 3 models + trials)
- At $0.077/hr ≈ **$0.50**; with setup/retries keep the cap at **$1.50**
- ~$2.50 stays as buffer for a second run if a model underperforms

## Cost-control built in

- `EarlyStopping` on every model (stops as soon as validation plateaus)
- Everything runs in one instance — zero repeat setup cost
- Quantized TFLite conversion is done on the same GPU box

## Next steps (after training)

1. Android app: CameraX -> feed frame to `seg.tflite` -> mask
2. OpenCV `findContours` on mask -> 4 corners -> `getPerspectiveTransform` + `warpPerspective`
3. Feed warped page to `enhance.tflite` -> clean output
4. Feed output to `cls.tflite` -> show document-type badge
5. (Optional) TFLite Metadata + TF Lite Task library for smoother integration