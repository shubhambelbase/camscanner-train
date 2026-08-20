#!/usr/bin/env bash
set -x
export DEBIAN_FRONTEND=noninteractive
: "${GITHUB_REPO:=shubhambelbase/camscanner-train}"
: "${VAST_API_KEY:?need VAST_API_KEY}"
: "${GH_TOKEN:?need GH_TOKEN}"

echo "=== [0] init: $(date -u) ==="

cd /root
if [ ! -d camscanner/.git ]; then
  git clone -q --depth 1 https://github.com/$GITHUB_REPO.git camscanner
fi
cd camscanner

echo "=== [1] install deps ==="
pip install -q -r docker/requirements.txt || pip install -q numpy opencv-python-headless Pillow tqdm midv500

echo "=== [2] download data ==="
python train/download_data.py --data /data || echo "[warn] data download partial; using synthetic fallbacks"

echo "=== [3] seg train ==="
python train/seg_train.py --data /data --out /out
echo "=== [4] enhance train ==="
python train/enhance_train.py --out /out
echo "=== [5] cls train ==="
python train/cls_train.py --data /data --out /out || echo "[warn] cls failed"
echo "=== [6] tflite ==="
python train/convert_tflite.py --data /data --out /out || echo "[warn] convert failed"

echo "=== [7] package + upload to GitHub Release ==="
cd /out
tar czf /results.tar.gz *.tflite 2>/dev/null
ls -la /results.tar.gz
API="https://api.github.com/repos/$GITHUB_REPO/releases"
curl -s -X POST -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  "$API" -d '{"tag_name":"v1","name":"Trained Models","body":"auto-built","draft":false,"prerelease":false}' > /tmp/release.json
UPLOAD_URL=$(python3 -c "import json;print(json.load(open('/tmp/release.json')).get('upload_url','').split('{')[0])" 2>/dev/null)
for f in /out/*.tflite; do
  echo "uploading $(basename $f)"
  curl -s -X POST -H "Authorization: token $GH_TOKEN" -H "Content-Type: application/octet-stream" \
    "$UPLOAD_URL?name=$(basename $f)" --data-binary "@$f" -o /dev/null || true
done
echo "RELEASE_URL: https://github.com/$GITHUB_REPO/releases/tag/v1"

echo "ALL_TRAINING_DONE"

echo "=== [8] self-destruct ==="
MY_IP=$(curl -s --max-time 10 https://api.ipify.org)
INST_ID=$(curl -s "https://console.vast.ai/api/v0/instances/?api_key=$VAST_API_KEY" | python3 -c "
import json,sys
try:
    data=json.load(sys.stdin)
    insts=data.get('instances', data if isinstance(data,list) else [])
    for i in insts:
        if i.get('public_ipaddr')==sys.argv[1]:
            print(i['id']); break
except Exception:
    pass
" "$MY_IP")
if [ -n "$INST_ID" ]; then
  echo "destroying instance $INST_ID"
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$INST_ID/" -H "Authorization: Bearer $VAST_API_KEY" || true
else
  echo "[warn] could not find instance id for IP $MY_IP; not self-destructing"
fi
echo "=== DONE ==="