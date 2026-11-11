"""SAM 3 (Segment Anything Model 3) + CVAT 2D Image Segmentation Toolkit.

Supports:
- Meta's SAM 3 via Hugging Face Transformers & Ultralytics
- Apple Silicon (M1/M2/M3/M4) MPS acceleration, CUDA, and CPU
- Concept-prompted segmentation (PCS), bounding boxes, points
- Semantic, Instance, and Panoptic post-processing
- CVAT COCO 1.0 (polygons/RLE) and Segmentation mask 1.1 dataset export/import
- Automated CVAT task synchronization via cvat-sdk
"""

from .cvat_client import CvatAutomationClient
from .cvat_exporter import (
    export_coco_zip,
    export_semantic_zip,
    mask_to_bbox,
    mask_to_polygons,
    mask_to_uncompressed_rle,
    save_visual_preview,
)
from .sam3_engine import (
    PredictedInstance,
    Sam3Segmenter,
    fill_mask_holes,
    mask_nms,
    resolve_instance_overlaps,
    select_device,
)

__all__ = [
    "Sam3Segmenter",
    "PredictedInstance",
    "CvatAutomationClient",
    "export_coco_zip",
    "export_semantic_zip",
    "save_visual_preview",
    "mask_to_polygons",
    "mask_to_uncompressed_rle",
    "mask_to_bbox",
    "fill_mask_holes",
    "mask_nms",
    "resolve_instance_overlaps",
    "select_device",
]

__version__ = "1.0.0"
