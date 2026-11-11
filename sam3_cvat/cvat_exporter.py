"""Export SAM 3 predictions to CVAT-compatible ZIPs and visual overlays.

Supports both contract formats required by Day 3:
  1. `Segmentation mask 1.1` (`labelmap.txt` + `SegmentationClass/<stem>.png`)
  2. `COCO 1.0` (`annotations/instances_default.json` with polygons or RLE)
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

try:
    from .sam3_engine import PredictedInstance
except ImportError:
    from sam3_engine import PredictedInstance


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.0015,
    min_area: float = 12.0,
) -> list[list[float]]:
    """Convert a binary HxW mask into a list of COCO polygon coordinate lists.

    Each polygon is `[x1, y1, x2, y2, ...]` with at least 6 coordinates (3 points).
    Preserves multiple disconnected components for occluded instances (`cp5_occlusion`).
    """
    import cv2

    uint_mask = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(uint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: list[list[float]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords: list[float] = []
        for x, y in pts:
            coords.extend([round(float(x), 2), round(float(y), 2)])
        if len(coords) >= 6:
            polygons.append(coords)

    return polygons


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """Convert a binary HxW mask to uncompressed COCO RLE (`{"size": [h, w], "counts": [...]}`)."""
    h, w = mask.shape
    flat = mask.T.ravel().astype(np.uint8)  # Fortran (column-major) order per COCO spec
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    diffs = np.diff(flat)
    change_indices = np.nonzero(diffs)[0] + 1
    splits = np.concatenate(([0], change_indices, [flat.size]))
    lengths = np.diff(splits).tolist()
    if flat[0] == 1:
        lengths = [0] + lengths
    return {"size": [int(h), int(w)], "counts": [int(x) for x in lengths]}


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    """Return COCO bounding box `[x, y, width, height]` from a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    return [
        round(x_min, 2),
        round(y_min, 2),
        round(x_max - x_min + 1.0, 2),
        round(y_max - y_min + 1.0, 2),
    ]


def build_labelmap_txt(classes: list[str], colors: dict[str, list[int]]) -> str:
    """Format `labelmap.txt` following CVAT Segmentation mask 1.1 specification."""
    lines = [
        "# label:color_rgb:parts:actions",
        "background:0,0,0::",
    ]
    for cls in classes:
        r, g, b = colors.get(cls, [128, 128, 128])
        lines.append(f"{cls}:{r},{g},{b}::")
    return "\n".join(lines) + "\n"


def export_semantic_zip(
    output_zip: Path,
    label_maps: dict[str, np.ndarray],
    classes: list[str],
    colors: dict[str, list[int]],
    class_to_id: dict[str, int],
) -> Path:
    """Write a CVAT `Segmentation mask 1.1` ZIP archive.

    Args:
        output_zip: Destination `.zip` path.
        label_maps: Mapping `{image_filename: HxW_int32_label_map}`.
        classes: Ordered class list from `classes.json`.
        colors: Mapping `{class_name: [R, G, B]}` from `classes.json`.
        class_to_id: Mapping `{class_name: int_id}` used in `label_maps`.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    id_to_rgb = {
        class_to_id[cls]: tuple(colors.get(cls, [128, 128, 128]))
        for cls in classes
    }

    stems: list[str] = []
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labelmap.txt", build_labelmap_txt(classes, colors))
        for filename, lmap in sorted(label_maps.items()):
            stem = Path(filename).stem
            stems.append(stem)
            h, w = lmap.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for cid, (r, g, b) in id_to_rgb.items():
                mask = lmap == cid
                rgb[mask] = (r, g, b)
            buf = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
            zf.writestr(f"SegmentationClass/{stem}.png", buf.getvalue())

        zf.writestr("ImageSets/Segmentation/default.txt", "\n".join(stems) + "\n")

    return output_zip


def export_coco_zip(
    output_zip: Path,
    predictions_by_image: dict[str, list[PredictedInstance]],
    image_sizes: dict[str, tuple[int, int]],
    classes: list[str],
    segmentation_format: Literal["polygon", "rle"] = "polygon",
) -> Path:
    """Write a CVAT `COCO 1.0` ZIP archive (`annotations/instances_default.json`).

    Args:
        output_zip: Destination `.zip` path.
        predictions_by_image: `{image_filename: [PredictedInstance, ...]}`.
        image_sizes: `{image_filename: (height, width)}`.
        classes: Ordered class names from `classes.json`.
        segmentation_format: `"polygon"` (recommended for editing in CVAT UI) or `"rle"`.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": idx, "name": cls, "supercategory": ""}
        for idx, cls in enumerate(classes, start=1)
    ]
    cls_to_cat_id = {c["name"]: c["id"] for c in categories}

    images_list = []
    annotations_list = []
    ann_id = 1

    for img_id, filename in enumerate(sorted(image_sizes.keys()), start=1):
        h, w = image_sizes[filename]
        images_list.append(
            {
                "id": img_id,
                "width": int(w),
                "height": int(h),
                "file_name": filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        for inst in predictions_by_image.get(filename, []):
            if inst.class_name not in cls_to_cat_id:
                continue
            area = float(inst.mask.sum())
            if area < 12:
                continue

            if segmentation_format == "polygon":
                seg = mask_to_polygons(inst.mask)
                if not seg:
                    continue
            else:
                seg = mask_to_uncompressed_rle(inst.mask)

            bbox = mask_to_bbox(inst.mask)
            annotations_list.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_to_cat_id[inst.class_name],
                    "segmentation": seg,
                    "area": round(area, 2),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False},
                }
            )
            ann_id += 1

    coco_payload = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "SAM3-CVAT-Tool", "version": "1.0"},
        "categories": categories,
        "images": images_list,
        "annotations": annotations_list,
    }

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "annotations/instances_default.json",
            json.dumps(coco_payload, ensure_ascii=False, indent=2) + "\n",
        )

    return output_zip


def save_visual_preview(
    image: Image.Image,
    output_path: Path,
    colors: dict[str, list[int]],
    semantic_map: np.ndarray | None = None,
    id_to_class: dict[int, str] | None = None,
    instances: list[PredictedInstance] | None = None,
    alpha: float = 0.50,
) -> Path:
    """Save a visual overlay image so learners can quickly QC masks before/with CVAT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orig = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = orig.copy()

    if semantic_map is not None and id_to_class is not None:
        for cid, cls_name in id_to_class.items():
            mask = semantic_map == cid
            if np.any(mask):
                rgb = np.array(colors.get(cls_name, [0, 255, 0]), dtype=np.float32)
                overlay[mask] = (1.0 - alpha) * orig[mask] + alpha * rgb
        vis_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    elif instances is not None:
        for inst in instances:
            mask = inst.mask
            if np.any(mask):
                rgb = np.array(colors.get(inst.class_name, [0, 255, 0]), dtype=np.float32)
                overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * rgb
        vis_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(vis_img)
        for idx, inst in enumerate(instances, start=1):
            bx, by, bw, bh = mask_to_bbox(inst.mask)
            if bw > 0 and bh > 0:
                color_tup = tuple(colors.get(inst.class_name, [255, 255, 0]))
                draw.rectangle([bx, by, bx + bw, by + bh], outline=color_tup, width=2)
                label_txt = f"#{idx} {inst.class_name} ({inst.score:.2f})"
                draw.text((bx + 3, max(0.0, by - 14)), label_txt, fill=(255, 255, 255))
    else:
        vis_img = image

    vis_img.save(output_path, quality=92)
    return output_path
