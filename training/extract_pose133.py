"""
Extract COCO-WholeBody-133 keypoints from WLASL clips with RTMW-x (rtmlib).
One .npy per video_id, shape (T, 133, 3) = (x, y, score) in pixel coords.
Signer = highest mean-confidence detected person per frame.

Run sharded across GPUs:
  for i in 0 1; do CUDA_VISIBLE_DEVICES=$i pose_venv/bin/python training/extract_pose133.py $i 2 & done
"""
import os, sys, glob, cv2, numpy as np
from rtmlib import Wholebody
from tqdm import tqdm

VID_DIR = os.environ.get('VID_DIR', 'data/wlasl-processed/videos')
OUT_DIR = os.environ.get('OUT_DIR', 'data/pose133')
os.makedirs(OUT_DIR, exist_ok=True)
shard, nshards = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)

model = Wholebody(mode='performance', backend='onnxruntime', device='cuda')
vids = sorted(glob.glob(f"{VID_DIR}/*.mp4"))[shard::nshards]
print(f"[shard {shard}/{nshards}] {len(vids)} videos -> {OUT_DIR}", flush=True)

done = 0
for vp in tqdm(vids, disable=(shard != 0)):
    vid = os.path.splitext(os.path.basename(vp))[0]
    out = f"{OUT_DIR}/{vid}.npy"
    if os.path.exists(out):
        done += 1; continue
    cap = cv2.VideoCapture(vp); frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts, scores = model(frame)          # (N,133,2), (N,133)
        if len(kpts) == 0:
            frames.append(np.zeros((133, 3), np.float32)); continue
        i = int(scores.mean(1).argmax())     # highest-conf person = signer
        frames.append(np.concatenate([kpts[i], scores[i][:, None]], 1).astype(np.float32))
    cap.release()
    arr = np.stack(frames) if frames else np.zeros((0, 133, 3), np.float32)
    np.save(out, arr); done += 1
    if shard == 0 and done % 200 == 0:
        print(f"  [shard0] {done}/{len(vids)}", flush=True)
print(f"[shard {shard}] DONE {done}/{len(vids)}", flush=True)
