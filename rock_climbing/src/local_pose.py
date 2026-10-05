"""ViTPose on this machine's GPU, answering in the gateway's own words.

`pose.request_poses` sends the clip to the VLM Run Gateway and gets back one
flat envelope of person records. :func:`run` produces the same envelope from
local weights, so everything downstream — `pose.unwrap`, the track choice, the
cache — reads it without knowing where it came from.

The gateway does three things in that one call, and so three things happen here:

  * **a person detector** finds the boxes. ViTPose is top-down: it is handed a
    crop of one person and places joints inside it, and has no idea where people
    are. RT-DETR does the finding.
  * **ViTPose+** places the 17 COCO joints in every box. The "plus" checkpoints
    carry one head per training dataset; index 0 is the COCO head.
  * **a tracker** gives each body a `track_id` that survives from frame to
    frame, which is what lets `pose.pick_track` choose the climber once for the
    whole clip. Boxes are matched to the previous frame's on overlap, and a
    track that goes unseen for a second is retired rather than revived onto
    somebody else.

torch and transformers are imported inside :func:`run`, so the gateway path
never needs them installed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton import KPT_NAMES

CONTENT_OBJECT_KPTS = "vid.pose.kpts"
COCO_HEAD = 0          # ViTPose+'s dataset index for the COCO-17 head
PERSON_LABEL = 0       # RT-DETR's COCO class id for a person


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in *a* and every box in *b*, both [x, y, w, h]."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def drop_nested(boxes: np.ndarray, scores: np.ndarray, *, max_nested: float) -> np.ndarray:
    """Keep one box per body where the detector returned several.

    RT-DETR has no suppression step of its own, and on a climber it sometimes
    answers twice: the whole body, and again just the torso or the half of them
    that is clear of the wall. The pair overlap too little to match as one box
    and so the second starts a track of its own — which then takes the climber's
    id the moment the first box flickers out. A box lying mostly inside a more
    confident one is that second answer, and is dropped.
    """
    order = np.argsort(-scores)
    kept: list[int] = []
    for i in order:
        x, y, w, h = boxes[i]
        nested = False
        for j in kept:
            X, Y, W, H = boxes[j]
            iw = min(x + w, X + W) - max(x, X)
            ih = min(y + h, Y + H) - max(y, Y)
            if iw > 0 and ih > 0 and iw * ih / max(min(w * h, W * H), 1e-9) > max_nested:
                nested = True
                break
        if not nested:
            kept.append(int(i))
    return boxes[sorted(kept)]


class Tracker:
    """Carry an id from frame to frame on box overlap.

    Deliberately simple. The demo needs the climber to keep one id up the wall
    and a passerby to get a different one; it does not need re-identification
    after a long occlusion, and `pose.pick_track` scores whole tracks against
    the route, so a climber split across two ids costs coverage, not correctness.
    """

    def __init__(self, *, min_iou: float, max_gap: int):
        self.min_iou = min_iou
        self.max_gap = max_gap
        self._boxes: dict[int, np.ndarray] = {}
        self._seen: dict[int, int] = {}
        self._next = 1

    def update(self, boxes: np.ndarray, frame: int) -> list[int]:
        for tid in [t for t, last in self._seen.items() if frame - last > self.max_gap]:
            del self._boxes[tid], self._seen[tid]

        ids = [-1] * len(boxes)
        live = list(self._boxes)
        if live and len(boxes):
            iou = _iou_matrix(np.stack([self._boxes[t] for t in live]), boxes)
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= self.min_iou:
                    ids[c] = live[r]
        for i, box in enumerate(boxes):
            if ids[i] < 0:
                ids[i] = self._next
                self._next += 1
            self._boxes[ids[i]] = box
            self._seen[ids[i]] = frame
        return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def run(video: Path, *, model: str, detector: str, extra_body: dict,
        det_threshold: float, max_nested: float, batch_size: int, track_min_iou: float,
        track_max_gap_seconds: float, half: bool = True, device: str | None = None,
        progress=None) -> tuple[dict, dict]:
    """Pose every decoded frame of *video*; return ``(payload, usage)``.

    *extra_body* is the gateway request's own (`pose.build_request`), read the
    way the gateway reads it: ``video_max_frames`` caps the frames decoded,
    ``video_fps`` is the detector's cadence — between detections the last boxes
    are reused, and pose still runs on every frame — and ``precision`` rounds
    the normalized coordinates.
    """
    import torch
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (half and device == "cuda") else torch.float32
    precision = int(extra_body.get("precision", 4))

    t_load = time.perf_counter()
    det_proc = AutoProcessor.from_pretrained(detector)
    det_model = RTDetrForObjectDetection.from_pretrained(detector, torch_dtype=dtype).to(device).eval()
    pose_proc = AutoProcessor.from_pretrained(model)
    pose_model = VitPoseForPoseEstimation.from_pretrained(model, torch_dtype=dtype).to(device).eval()
    load_seconds = time.perf_counter() - t_load

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = extra_body.get("video_max_frames")
    n_target = min(total, int(limit)) if limit else total
    det_stride = max(1, round(fps / float(extra_body.get("video_fps") or fps)))

    tracker = Tracker(min_iou=track_min_iou,
                      max_gap=max(1, round(track_max_gap_seconds * fps)))
    scale = np.array([width, height], dtype=np.float64)
    items: list[dict] = []
    frames: list[dict] = []
    last_boxes = np.zeros((0, 4), dtype=np.float32)

    def detect(images: list[np.ndarray]) -> list[np.ndarray]:
        """Person boxes per image, as pixel [x, y, w, h]."""
        inputs = det_proc(images=images, return_tensors="pt").to(device, dtype)
        outputs = det_model(**inputs)
        sizes = torch.tensor([[height, width]] * len(images), device=device)
        results = det_proc.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=det_threshold)
        out = []
        for res in results:
            people = res["labels"] == PERSON_LABEL
            boxes = res["boxes"][people].float().cpu().numpy()
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            out.append(drop_nested(boxes, res["scores"][people].float().cpu().numpy(),
                                   max_nested=max_nested))
        return out

    def flush(indices: list[int], images: list[np.ndarray]) -> None:
        nonlocal last_boxes
        due = [k for k, idx in enumerate(indices) if idx % det_stride == 0]
        found = dict(zip(due, detect([images[k] for k in due]))) if due else {}
        per_frame = []
        for k in range(len(indices)):
            if k in found:
                last_boxes = found[k]
            per_frame.append(last_boxes)

        posed = [k for k in range(len(indices)) if len(per_frame[k])]
        results = []
        if posed:
            boxes = [per_frame[k] for k in posed]
            inputs = pose_proc([images[k] for k in posed], boxes=boxes,
                               return_tensors="pt").to(device, dtype)
            n_crops = inputs["pixel_values"].shape[0]
            outputs = pose_model(**inputs, dataset_index=torch.full(
                (n_crops,), COCO_HEAD, device=device, dtype=torch.long))
            outputs.heatmaps = outputs.heatmaps.float()
            results = pose_proc.post_process_pose_estimation(outputs, boxes=boxes)

        by_frame = dict(zip(posed, results))
        for k, idx in enumerate(indices):
            frames.append({"frame_id": idx, "frame_ts": round(idx / fps, 3)})
            people = by_frame.get(k)
            if not people:
                tracker.update(np.zeros((0, 4)), idx)
                continue
            ids = tracker.update(per_frame[k], idx)
            for box, person, tid in zip(per_frame[k], people, ids):
                xy = person["keypoints"].float().cpu().numpy() / scale
                items.append({
                    "bbox_xywh": [round(float(v), precision) for v in
                                  (box[0] / width, box[1] / height,
                                   box[2] / width, box[3] / height)],
                    "label": "person",
                    "kpts_xy": [[round(float(x), precision), round(float(y), precision)]
                                for x, y in xy],
                    "kpts_score": [round(float(s), precision)
                                   for s in person["scores"].float().cpu().numpy()],
                    "frame_id": idx,
                    "track_id": tid,
                })

    t_run = time.perf_counter()
    indices: list[int] = []
    images: list[np.ndarray] = []
    with torch.inference_mode():
        for idx in range(n_target):
            ok, frame = cap.read()
            if not ok:
                break
            indices.append(idx)
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(indices) == batch_size:
                flush(indices, images)
                indices, images = [], []
                if progress:
                    progress(idx + 1, n_target)
        if indices:
            flush(indices, images)
            if progress:
                progress(n_target, n_target)
    cap.release()
    infer_seconds = time.perf_counter() - t_run

    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
    del det_model, pose_model
    if device == "cuda":
        torch.cuda.empty_cache()

    n = len(frames)
    payload = {
        "model": model,
        "method": "pose",
        "video_hash": _sha256(video),
        "video_width": width,
        "video_height": height,
        "video_fps": float(fps),
        "video_nframes": n,
        "video_duration": round(n / fps, 3),
        "content": {
            "object": CONTENT_OBJECT_KPTS,
            "items": items,
            "frames": frames,
            "kpts_labels": list(KPT_NAMES),
        },
    }
    usage = {
        "backend": "local",
        "device": (torch.cuda.get_device_name(0) if device == "cuda" else device),
        "detector": detector,
        "dtype": str(dtype).removeprefix("torch."),
        "frames": n,
        "load_seconds": round(load_seconds, 1),
        "infer_seconds": round(infer_seconds, 1),
        "fps": round(n / infer_seconds, 1) if infer_seconds else None,
        "peak_vram_mb": round(peak_mb) if peak_mb is not None else None,
        "cost": 0.0,
    }
    return payload, usage
