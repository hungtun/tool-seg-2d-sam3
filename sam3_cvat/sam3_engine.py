"""SAM 3 Inference Engine & Task-Specific Post-Processing (Hugging Face Transformers).

Supports Meta's Segment Anything Model 3 (`facebook/sam3`) using Promptable Concept
Segmentation (PCS) to segment all classes defined in each Day 3 task's `classes.json`.
Optimized for Apple Silicon (`mps`), CUDA, and CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Enable MPS fallback for any unsupported PyTorch operators on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import numpy as np
from PIL import Image


# Priority order when painting semantic layers (from background to foreground).
# Smaller/thinner foreground objects are painted last so they are not overwritten.
SEMANTIC_PAINT_ORDER: list[str] = [
    "sky",
    "building",
    "vegetation",
    "road",
    "sidewalk",
    "truck",
    "bus",
    "car",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "traffic sign",
    "pole",
]

# Concept prompt aliases to improve SAM 3 PCS recall on street-scene classes
# Note: Never use "pavement" for sidewalk because in US English "pavement" means asphalt road!
CONCEPT_PROMPTS: dict[str, list[str]] = {
    "road": ["road"],
    "sidewalk": ["sidewalk"],
    "building": ["building"],
    "vegetation": ["vegetation", "trees and bushes"],
    "sky": ["sky"],
    "person": ["person"],
    "bicycle": ["bicycle"],
    "car": ["car"],
    "motorcycle": ["motorcycle"],
    "bus": ["bus"],
    "truck": ["truck"],
    "traffic light": ["traffic light"],
    "traffic sign": ["traffic sign", "street sign"],
    "pole": ["pole", "street light pole"],
}

# Per-class default confidence thresholds
CLASS_THRESHOLDS: dict[str, float] = {
    "pole": 0.25,
    "traffic sign": 0.28,
    "traffic light": 0.28,
    "bicycle": 0.35,
    "motorcycle": 0.35,
    "sidewalk": 0.35,
    "road": 0.30,
}


@dataclass
class PredictedInstance:
    """Single segmented object or stuff region."""

    class_name: str
    mask: np.ndarray  # HxW bool array
    score: float
    box: list[float] = field(default_factory=list)  # [x1, y1, x2, y2]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def select_device(preferred: str | None = None) -> str:
    """Detect best available torch device (mps -> cuda -> cpu)."""
    import torch

    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes inside each connected component of a binary mask.

    Essential for `cp1_holes` (windows/windshields inside vehicles stay inside mask).
    """
    try:
        from scipy.ndimage import binary_fill_holes

        return binary_fill_holes(mask).astype(bool)
    except ImportError:
        import cv2

        uint_mask = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(uint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(uint_mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary mask IoU."""
    inter = int((a & b).sum())
    if inter == 0:
        return 0.0
    union = int((a | b).sum())
    return inter / union if union else 0.0


def mask_nms(
    instances: list[PredictedInstance],
    iou_threshold: float = 0.65,
    containment_threshold: float = 0.85,
) -> list[PredictedInstance]:
    """Non-Maximum Suppression over binary masks within the same class.

    Prevents duplicate masks on the same vehicle while keeping adjacent vehicles
    separate (`cp2_slice`).
    """
    kept: list[PredictedInstance] = []
    by_class: dict[str, list[PredictedInstance]] = {}
    for inst in instances:
        by_class.setdefault(inst.class_name, []).append(inst)

    for cls, group in by_class.items():
        # Sort by confidence score descending, then area descending
        group.sort(key=lambda x: (x.score, x.area), reverse=True)
        cls_kept: list[PredictedInstance] = []
        for cand in group:
            if cand.area < 16:
                continue
            duplicate = False
            for existing in cls_kept:
                iou = mask_iou(cand.mask, existing.mask)
                if iou >= iou_threshold:
                    duplicate = True
                    break
                # Also suppress if candidate is almost completely inside an existing mask of same class
                inter = int((cand.mask & existing.mask).sum())
                if cand.area > 0 and (inter / cand.area) >= containment_threshold:
                    duplicate = True
                    break
            if not duplicate:
                cls_kept.append(cand)
        kept.extend(cls_kept)
    return kept


def resolve_instance_overlaps(instances: list[PredictedInstance]) -> list[PredictedInstance]:
    """Ensure two instances do not claim the exact same boundary pixels (`cp2_slice`).

    Higher-confidence masks keep contested pixels; fragments that become too small
    are dropped.
    """
    sorted_inst = sorted(instances, key=lambda x: x.score, reverse=True)
    occupied = np.zeros_like(sorted_inst[0].mask, dtype=bool) if sorted_inst else None
    resolved: list[PredictedInstance] = []
    for inst in sorted_inst:
        clean_mask = inst.mask & (~occupied)
        if int(clean_mask.sum()) < 25:
            continue
        occupied |= clean_mask
        resolved.append(
            PredictedInstance(
                class_name=inst.class_name,
                mask=clean_mask,
                score=inst.score,
                box=inst.box,
            )
        )
    return resolved


def fill_unlabeled_nearest(label_map: np.ndarray, ignore_val: int = 255) -> np.ndarray:
    """Fill unlabeled (255) pixels using nearest labeled pixel (`cp6_coverage`)."""
    unlabeled = label_map == ignore_val
    if not np.any(unlabeled) or np.all(unlabeled):
        return label_map
    try:
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(unlabeled, return_indices=True)
        filled = label_map.copy()
        filled[unlabeled] = label_map[tuple(indices[:, unlabeled])]
        return filled
    except ImportError:
        return label_map


class Sam3Segmenter:
    """Unified SAM 3 wrapper supporting both Hugging Face `transformers` and `ultralytics`."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        backend: str = "auto",
        device: str | None = None,
        default_threshold: float = 0.35,
        mask_threshold: float = 0.5,
        use_prompt_aliases: bool = True,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self.device = select_device(device)
        self.default_threshold = default_threshold
        self.mask_threshold = mask_threshold
        self.use_prompt_aliases = use_prompt_aliases
        self._model: Any = None
        self._processor: Any = None
        self._ultra_predictor: Any = None
        self._active_backend: str | None = None

    def _resolve_backend(self) -> str:
        if self._active_backend is not None:
            return self._active_backend
        if self.backend in ("transformers", "ultralytics"):
            self._active_backend = self.backend
            return self._active_backend
        # If model_id ends with .pt, use ultralytics
        if str(self.model_id).endswith(".pt"):
            self._active_backend = "ultralytics"
            return self._active_backend
        try:
            from transformers import Sam3Model, Sam3Processor  # noqa: F401

            self._active_backend = "transformers"
        except ImportError:
            self._active_backend = "ultralytics"
        return self._active_backend

    def _lazy_load(self) -> None:
        backend = self._resolve_backend()
        if backend == "transformers":
            if self._model is not None and self._processor is not None:
                return
            import torch
            from transformers import Sam3Model, Sam3Processor

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Build candidate repo list: try requested model_id first, then ungated mirrors
            candidates = [self.model_id]
            for mirror in ("tkclam/sam3", "Justin331/sam3", "unsloth/sam3.1"):
                if mirror not in candidates:
                    candidates.append(mirror)

            last_err: Exception | None = None
            for candidate_id in candidates:
                try:
                    print(
                        f"[SAM3-HF] Đang tải model '{candidate_id}' trên thiết bị '{self.device}'..."
                    )
                    self._processor = Sam3Processor.from_pretrained(candidate_id, token=hf_token)
                    self._model = Sam3Model.from_pretrained(
                        candidate_id,
                        torch_dtype=dtype,
                        token=hf_token,
                    ).to(self.device)
                    self._model.eval()
                    self.model_id = candidate_id
                    print(f"[SAM3-HF] Tải model '{candidate_id}' hoàn tất ({self.device}).")
                    break
                except Exception as exc:
                    last_err = exc
                    msg = str(exc).lower()
                    if "gated" in msg or "401" in msg or "403" in msg or "not found" in msg or "unauthorized" in msg:
                        print(
                            f"[SAM3-HF] Repo '{candidate_id}' yêu cầu quyền truy cập (Gated/401). "
                            f"Đang tự động chuyển sang bản mirror không khóa (ungated)..."
                        )
                        continue
                    raise
            else:
                if last_err is not None:
                    raise last_err
        else:
            if self._ultra_predictor is not None:
                return
            from ultralytics.models.sam import SAM3SemanticPredictor

            ckpt = self.model_id if str(self.model_id).endswith(".pt") else "sam3.pt"
            print(
                f"[SAM3-Ultralytics] Đang khởi tạo SAM3SemanticPredictor với checkpoint '{ckpt}' trên '{self.device}'..."
            )
            overrides = {
                "conf": self.default_threshold,
                "task": "segment",
                "mode": "predict",
                "model": ckpt,
                "device": self.device,
                "half": self.device == "cuda",
                "verbose": False,
            }
            self._ultra_predictor = SAM3SemanticPredictor(overrides=overrides)
            self._ultra_predictor.setup_model()
            print(f"[SAM3-Ultralytics] Khởi tạo model hoàn tất ({self.device}).")

    def query_concept(
        self,
        image: Image.Image,
        concept_text: str,
        threshold: float,
    ) -> list[tuple[np.ndarray, float, list[float]]]:
        """Run SAM 3 text-prompted concept segmentation for a single phrase."""
        self._lazy_load()
        backend = self._resolve_backend()

        if backend == "ultralytics":
            self._ultra_predictor.args.conf = threshold
            img_bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
            results = self._ultra_predictor(source=img_bgr, text=[concept_text])
            out: list[tuple[np.ndarray, float, list[float]]] = []
            if not results or results[0].masks is None:
                return out
            res = results[0]
            masks_np = res.masks.data.detach().cpu().numpy().astype(bool)
            boxes_data = res.boxes
            scores_np = (
                boxes_data.conf.detach().cpu().numpy().astype(float)
                if boxes_data is not None and boxes_data.conf is not None
                else np.ones(len(masks_np), dtype=float)
            )
            xyxy_np = (
                boxes_data.xyxy.detach().cpu().numpy().astype(float)
                if boxes_data is not None and boxes_data.xyxy is not None
                else np.zeros((len(masks_np), 4), dtype=float)
            )
            for m, s, b in zip(masks_np, scores_np, xyxy_np):
                if float(s) >= threshold and int(m.sum()) > 0:
                    out.append((m, float(s), b.tolist()))
            return out

        import torch

        inputs = self._processor(
            images=image,
            text=concept_text,
            return_tensors="pt",
        )
        # Move tensor inputs to device
        inputs_on_device = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self._model(**inputs_on_device)

        orig_sizes = inputs.get("original_sizes")
        if orig_sizes is not None and hasattr(orig_sizes, "tolist"):
            target_sizes = orig_sizes.tolist()
        else:
            target_sizes = [(image.height, image.width)]

        results = self._processor.post_process_instance_segmentation(
            outputs=outputs,
            threshold=threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]

        masks_t = results.get("masks")
        scores_t = results.get("scores")
        boxes_t = results.get("boxes")

        out = []
        if masks_t is None or len(masks_t) == 0:
            return out

        masks_np = masks_t.detach().cpu().numpy().astype(bool)
        scores_np = scores_t.detach().cpu().numpy().astype(float)
        boxes_np = (
            boxes_t.detach().cpu().numpy().astype(float)
            if boxes_t is not None
            else np.zeros((len(masks_np), 4))
        )

        for m, s, b in zip(masks_np, scores_np, boxes_np):
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            if int(m.sum()) > 0:
                out.append((m, float(s), b.tolist()))
        return out

    def detect_classes(
        self,
        image: Image.Image,
        classes: list[str],
        task_name: str = "",
    ) -> list[PredictedInstance]:
        """Run SAM 3 across all target classes in `classes.json`."""
        raw_instances: list[PredictedInstance] = []

        for cls in classes:
            thr = CLASS_THRESHOLDS.get(cls, self.default_threshold)
            if task_name == "cp3_thin" and cls in ("pole", "traffic sign"):
                thr = min(thr, 0.25)
            elif task_name in ("cp2_slice", "cp5_occlusion") and cls in ("car", "person", "motorcycle", "bicycle"):
                thr = max(thr, 0.42)

            prompts = (
                CONCEPT_PROMPTS.get(cls, [cls])
                if self.use_prompt_aliases
                else [cls]
            )

            for prompt in prompts:
                detections = self.query_concept(image, prompt, threshold=thr)
                for mask, score, box in detections:
                    # Fill holes inside vehicles or when task is cp1_holes
                    if task_name == "cp1_holes" or cls in ("car", "bus", "truck"):
                        mask = fill_mask_holes(mask)
                    raw_instances.append(
                        PredictedInstance(
                            class_name=cls,
                            mask=mask,
                            score=score,
                            box=box,
                        )
                    )

        # Deduplicate masks per class via Mask NMS
        nms_iou = 0.55 if task_name in ("cp2_slice", "cp5_occlusion") else 0.65
        filtered = mask_nms(raw_instances, iou_threshold=nms_iou)
        return filtered

    def predict_semantic(
        self,
        image: Image.Image,
        classes: list[str],
        class_to_id: dict[str, int],
        task_name: str = "",
        fill_gaps: bool = False,
    ) -> np.ndarray:
        """Produce an HxW int32 semantic label map (class index in `classes`, or 255 for unlabeled)."""
        h, w = image.height, image.width
        instances = self.detect_classes(image, classes, task_name=task_name)

        stuff_set = {"sky", "building", "vegetation", "road", "sidewalk"}
        label_map = np.full((h, w), 255, dtype=np.int32)
        best_stuff_score = np.full((h, w), -1.0, dtype=np.float32)

        # 1. Resolve stuff classes per pixel by highest confidence score
        for inst in instances:
            if inst.class_name not in class_to_id or inst.class_name not in stuff_set:
                continue
            cid = class_to_id[inst.class_name]
            better = inst.mask & (inst.score > best_stuff_score)
            label_map[better] = cid
            best_stuff_score[better] = float(inst.score)

        # 2. Paint foreground / thin classes on top in priority order
        fg_instances = [
            inst for inst in instances
            if inst.class_name in class_to_id and inst.class_name not in stuff_set
        ]
        fg_instances.sort(
            key=lambda inst: (
                SEMANTIC_PAINT_ORDER.index(inst.class_name)
                if inst.class_name in SEMANTIC_PAINT_ORDER
                else len(SEMANTIC_PAINT_ORDER),
                inst.score,
            )
        )
        for inst in fg_instances:
            cid = class_to_id[inst.class_name]
            label_map[inst.mask] = cid

        if fill_gaps:
            label_map = fill_unlabeled_nearest(label_map, ignore_val=255)

        return label_map

    def predict_instances(
        self,
        image: Image.Image,
        classes: list[str],
        task_name: str = "",
    ) -> list[PredictedInstance]:
        """Produce non-overlapping instance predictions for COCO 1.0 tasks."""
        instances = self.detect_classes(image, classes, task_name=task_name)
        if task_name == "cp2_slice":
            instances = resolve_instance_overlaps(instances)
        return instances

    def predict_panoptic(
        self,
        image: Image.Image,
        classes: list[str],
        stuff_classes: list[str],
        task_name: str = "hard_panoptic",
    ) -> list[PredictedInstance]:
        """Produce merged `stuff` masks + distinct `thing` masks for COCO 1.0 panoptic export.

        Note: `lab_utils.instances_to_panoptic` paints masks sorted from largest area
        to smallest area, merging `stuff` by class and keeping `thing` separate.
        """
        raw = self.detect_classes(image, classes, task_name=task_name)
        stuff_set = set(stuff_classes)

        stuff_merged: dict[str, np.ndarray] = {}
        stuff_scores: dict[str, list[float]] = {}
        things: list[PredictedInstance] = []

        for inst in raw:
            if inst.class_name in stuff_set:
                if inst.class_name not in stuff_merged:
                    stuff_merged[inst.class_name] = inst.mask.copy()
                    stuff_scores[inst.class_name] = [inst.score]
                else:
                    stuff_merged[inst.class_name] |= inst.mask
                    stuff_scores[inst.class_name].append(inst.score)
            else:
                things.append(inst)

        # Resolve overlaps among things so adjacent cars/people don't merge or collide
        things = resolve_instance_overlaps(things)

        # Resolve overlaps among stuff classes using SEMANTIC_PAINT_ORDER
        h, w = image.height, image.width
        stuff_canvas = np.full((h, w), "", dtype=object)
        ordered_stuff = sorted(
            stuff_merged.keys(),
            key=lambda c: SEMANTIC_PAINT_ORDER.index(c) if c in SEMANTIC_PAINT_ORDER else 99,
        )
        for cls in ordered_stuff:
            stuff_canvas[stuff_merged[cls]] = cls

        final_list: list[PredictedInstance] = []
        for cls in ordered_stuff:
            m = stuff_canvas == cls
            if int(m.sum()) >= 64:
                final_list.append(
                    PredictedInstance(
                        class_name=cls,
                        mask=m,
                        score=float(np.mean(stuff_scores[cls])),
                    )
                )

        final_list.extend(things)
        return final_list
