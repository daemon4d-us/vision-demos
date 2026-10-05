"""SAM 3 on this machine's GPU, answering in the gateway's own words.

The gateway is asked three things of SAM, and each has a counterpart here that
returns the same envelope, so `holds.observations`, the floor consensus and the
hold recovery read a local reply without knowing where it came from:

  * :func:`track` — the route. A text prompt, tracked through sampled frames of
    the clip, one `track_id` per hold (`Sam3VideoModel`).
  * :meth:`Local.segment` — the floor. A text prompt on one still (`Sam3Model`).
  * :meth:`Local.segment_box` — a missed hold. A box on one still, used as a
    visual exemplar, so what comes back is a proposal set over the image steered
    by the box — which is what `recover._pick` is written against.

The checkpoint is `facebook/sam3`, not the gateway's SAM 3.1: 3.1 ships as a raw
checkpoint for Meta's own package, and Transformers loads SAM 3. The envelope's
`model` field says which one answered.

Only one of the two models is held on the GPU at a time. They are the same
weights wearing different heads, each a couple of GB in half precision, and the
video model's memory grows with every frame it tracks — an 8 GB card has room
for the tracker's working set or for a second model, not both.

torch and transformers are imported lazily, so the gateway path never needs
them installed.
"""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

CONTENT_OBJECT_VIDEO = "vid.segment.masks"
CONTENT_OBJECT_IMAGE = "img.segment.masks"

_loaded: dict = {}     # {"kind": "video" | "image", "model": ..., "processor": ...}


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(half: bool):
    import torch
    return torch.float16 if (half and _device() == "cuda") else torch.float32


def _load(kind: str, checkpoint: str, half: bool):
    """The requested model, evicting the other one first."""
    import torch
    from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor

    key = (kind, checkpoint, half)
    if _loaded.get("key") == key:
        return _loaded["model"], _loaded["processor"]
    unload()
    model_cls, proc_cls = ((Sam3VideoModel, Sam3VideoProcessor) if kind == "video"
                           else (Sam3Model, Sam3Processor))
    model = model_cls.from_pretrained(checkpoint, torch_dtype=_dtype(half)).to(_device()).eval()
    processor = proc_cls.from_pretrained(checkpoint)
    _loaded.update(key=key, model=model, processor=processor)
    return model, processor


def unload() -> None:
    """Drop whatever is on the GPU, so the next model (or ViTPose) has the room."""
    import torch
    _loaded.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── the wire format ──────────────────────────────────────────────────────────

def _png_label_map(labels: np.ndarray) -> dict:
    ok, buf = cv2.imencode(".png", labels)
    if not ok:
        raise RuntimeError("cv2.imencode failed on a label map")
    return {"format": "png", "height": int(labels.shape[0]), "width": int(labels.shape[1]),
            "data": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")}


def _polygons(mask: np.ndarray, precision: int) -> list[list[list[float]]]:
    """A mask's outlines as simplified rings, normalized — the gateway's `polys_xy`."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) >= 3:
            rings.append([[round(float(x) / w, precision), round(float(y) / h, precision)]
                          for x, y in approx.reshape(-1, 2)])
    return rings


def _instance(mask: np.ndarray, score: float, label: str, precision: int) -> dict | None:
    """One mask as a gateway item row, with the box taken off the mask itself."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return {
        "bbox_xywh": [round(float(v), precision) for v in
                      (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)],
        "label": label,
        "score": round(float(score), precision),
        "area": round(float(len(xs)) / (h * w), precision),
        "polys_xy": _polygons(mask, precision),
    }


def _stack_labels(masks: list[np.ndarray], ids: list[int], scores: list[float],
                  shape: tuple[int, int]) -> np.ndarray:
    """One uint8 label map from per-instance masks; the surer instance wins a pixel."""
    labels = np.zeros(shape, dtype=np.uint8)
    for k in np.argsort(scores):          # ascending, so the best is painted last
        labels[masks[k]] = ids[k]
    return labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


# ── the route: text-prompted tracking ────────────────────────────────────────

def track(video: Path, *, checkpoint: str, prompt: str, start_frame: int, n_frames: int,
          skip_frames: int, max_frames: int, half: bool = True, precision: int = 4,
          progress=None) -> tuple[dict, dict]:
    """Track every instance of *prompt* through one stretch of *video*.

    The stretch is ``[start_frame, start_frame + n_frames)``, read straight out
    of the clip — the gateway needs a segment cut into its own file to upload,
    and this does not. Every ``skip_frames``-th frame is looked at, at most
    ``max_frames`` of them, which is the gateway's `track` sampling.

    `frame_id` and `track_id` both count from this stretch's own start, exactly
    as a gateway reply to an uploaded segment does, so `holds.shift` and
    `holds.concat` stitch local parts the same way.

    The sampled frames stay in system memory and are fed to the GPU one at a
    time; what grows on the GPU is the tracker's own memory of each frame.
    """
    import torch

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = set(range(0, n_frames, max(1, skip_frames))[:max_frames])
    images: list[np.ndarray] = []
    local_ids: list[int] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    for offset in range(n_frames):          # sequential: seeking per sample is slower
        ok, frame = capture.read() if offset in wanted else (capture.grab(), None)
        if not ok:
            break
        if offset in wanted:
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            local_ids.append(offset)
    capture.release()
    if not images:
        raise RuntimeError(f"No frames read from {video} at {start_frame}+{n_frames}")

    t_load = time.perf_counter()
    model, processor = _load("video", checkpoint, half)
    load_seconds = time.perf_counter() - t_load
    device = _device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_run = time.perf_counter()
    session = processor.init_video_session(
        video=images, inference_device=device, processing_device="cpu",
        video_storage_device="cpu", dtype=_dtype(half))
    session = processor.add_text_prompt(session, prompt)

    items: list[dict] = []
    frames: dict[int, dict] = {}
    with torch.inference_mode():
        for output in model.propagate_in_video_iterator(session):
            result = processor.postprocess_outputs(session, output)
            frame_id = local_ids[output.frame_idx]
            masks = result["masks"].cpu().numpy()
            scores = result["scores"].float().cpu().numpy().tolist()
            # SAM numbers objects from 0 and the label map spends 0 on "nothing".
            ids = [int(i) + 1 for i in result["object_ids"].tolist()]
            kept_masks, kept_ids, kept_scores = [], [], []
            for mask, score, track_id in zip(masks, scores, ids):
                row = _instance(mask, score, prompt, precision)
                if row is None or track_id > 255:
                    continue
                items.append({**row, "frame_id": frame_id, "track_id": track_id})
                kept_masks.append(mask); kept_ids.append(track_id); kept_scores.append(score)
            frames[frame_id] = {
                "frame_id": frame_id, "frame_ts": round(frame_id / fps, 3),
                "mask": _png_label_map(_stack_labels(kept_masks, kept_ids, kept_scores,
                                                     (height, width)))}
            if progress:
                progress(len(frames), len(images))
    infer_seconds = time.perf_counter() - t_run
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    del session
    if device == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "model": checkpoint, "method": "track",
        "video_width": width, "video_height": height, "video_fps": float(fps),
        "video_nframes": int(n_frames), "video_duration": round(n_frames / fps, 3),
        "content": {"object": CONTENT_OBJECT_VIDEO, "items": items,
                    "frames": [frames[k] for k in sorted(frames)]},
    }
    usage = {"backend": "local", "frames": len(images),
             "load_seconds": round(load_seconds, 1), "infer_seconds": round(infer_seconds, 1),
             "peak_vram_mb": round(peak_mb) if peak_mb is not None else None, "cost": 0.0}
    return payload, usage


# ── stills: the floor, and a missed hold ─────────────────────────────────────

class Local:
    """Stands where the gateway's client does, for the two still-image calls.

    `holds.segment_frames` and `recover._segment_box` are handed a client and
    make a request with it. Handed one of these instead, they call it directly.
    """

    def __init__(self, *, checkpoint: str, half: bool = True, threshold: float = 0.3,
                 precision: int = 4):
        self.checkpoint = checkpoint
        self.half = half
        self.threshold = threshold
        self.precision = precision

    def _run(self, image_bgr: np.ndarray, label: str, **prompt) -> dict:
        import torch

        model, processor = _load("image", self.checkpoint, self.half)
        height, width = image_bgr.shape[:2]
        inputs = processor(images=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                           return_tensors="pt", **prompt).to(_device())
        inputs["pixel_values"] = inputs["pixel_values"].to(_dtype(self.half))
        if "input_boxes" in inputs:
            inputs["input_boxes"] = inputs["input_boxes"].to(_dtype(self.half))
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=self.threshold, mask_threshold=0.5,
            target_sizes=[(height, width)])[0]
        masks = result["masks"].bool().cpu().numpy()
        scores = result["scores"].float().cpu().numpy().tolist()

        items, kept_masks, kept_ids, kept_scores = [], [], [], []
        for mask, score in sorted(zip(masks, scores), key=lambda pair: -pair[1])[:255]:
            row = _instance(mask, score, label, self.precision)
            if row is None:
                continue
            instance_id = len(items) + 1
            items.append({**row, "instance_id": instance_id})
            kept_masks.append(mask); kept_ids.append(instance_id); kept_scores.append(score)
        return {
            "model": self.checkpoint,
            "content": {"object": CONTENT_OBJECT_IMAGE, "items": items,
                        "mask": _png_label_map(_stack_labels(
                            kept_masks, kept_ids, kept_scores, (height, width)))},
        }

    def segment(self, image_bgr: np.ndarray, prompt: str) -> dict:
        """Every instance of *prompt* in one still."""
        return self._run(image_bgr, prompt, text=prompt)

    def segment_box(self, image_bgr: np.ndarray, bbox_xywh) -> dict:
        """Everything in the still that looks like what is inside *bbox_xywh*.

        The box is normalized [x, y, w, h], as the gateway takes it.
        """
        height, width = image_bgr.shape[:2]
        x, y, w, h = (float(v) for v in bbox_xywh)
        box = [x * width, y * height, (x + w) * width, (y + h) * height]
        return self._run(image_bgr, "object", input_boxes=[[box]],
                         input_boxes_labels=[[1]])
