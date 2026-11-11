"""Command-Line Interface for SAM 3 + CVAT 2D Image Segmentation Toolkit.

Supports:
  1. Segmenting a single image with text prompts (`predict-image`)
  2. Segmenting a whole folder of images into COCO 1.0 format (`predict-folder`)
  3. Automated segmentation across dataset tasks from a manifest (`predict`)
  4. Synchronizing tasks and pre-annotations with CVAT Local (`cvat-push`, `cvat-pull`)

Examples:
  # 1. Segment a single image:
  python cli.py predict-image --image photo.jpg --prompts "person" "car" --output result.jpg

  # 2. Segment all tasks in a dataset:
  python cli.py predict --manifest data/manifest.json --all

  # 3. Push tasks and pre-annotations to CVAT Local:
  python cli.py cvat-push --manifest data/manifest.json --all --host http://localhost:8080 -u admin -p admin

  # 4. Pull finished annotations from CVAT:
  python cli.py cvat-pull --manifest data/manifest.json --all --host http://localhost:8080 -u admin -p admin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sam3_cvat.cvat_client import CvatAutomationClient
from sam3_cvat.cvat_exporter import (
    export_coco_zip,
    export_semantic_zip,
    save_visual_preview,
)
from sam3_cvat.sam3_engine import PredictedInstance, Sam3Segmenter


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data.get("tasks", data)


def load_task_metadata(task_name: str, task_info: dict[str, Any], data_root: Path) -> dict[str, Any]:
    rel_path = task_info.get("path", task_name)
    base_dir = data_root / rel_path if (data_root / rel_path).exists() else data_root / task_name
    classes_meta_path = base_dir / "classes.json"
    if not classes_meta_path.is_file():
        raise FileNotFoundError(f"classes.json not found in {base_dir}")
    classes_meta = json.loads(classes_meta_path.read_text(encoding="utf-8"))
    images = sorted((base_dir / "images").glob("*.jpg"))
    if not images:
        images = sorted(base_dir.glob("*.jpg")) + sorted(base_dir.glob("*.png"))

    return {
        "name": task_name,
        "type": task_info.get("type", "instance"),
        "weight": task_info.get("weight", 0),
        "edge": task_info.get("edge", ""),
        "base_dir": base_dir,
        "images": images,
        "classes": classes_meta.get("classes", []),
        "colors": classes_meta.get("colors", {}),
        "stuff": classes_meta.get("stuff", []),
        "cvat_labels_path": base_dir / "cvat-labels.json",
    }


def run_predict_for_task(
    segmenter: Sam3Segmenter,
    meta: dict[str, Any],
    out_dir: Path,
    vis_dir: Path,
    mask_format: str = "polygon",
    fill_gaps: bool = False,
) -> dict[str, Any]:
    from PIL import Image

    task_name = meta["name"]
    task_type = meta["type"]
    classes: list[str] = meta["classes"]
    colors: dict[str, list[int]] = meta["colors"]
    images: list[Path] = meta["images"]

    print(f"\n============================================================")
    print(f"[Task: {task_name}] Type: {task_type.upper()} | Images: {len(images)} | Classes: {classes}")
    if meta.get("edge"):
        print(f"[Rule/Constraint] {meta['edge']}")
    print(f"============================================================")

    zip_path = out_dir / f"{task_name}.zip"
    task_summary: dict[str, Any] = {
        "task": task_name,
        "type": task_type,
        "zip_path": str(zip_path),
        "images": {},
    }

    if task_type == "semantic":
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}
        id_to_class = {idx: cls for cls, idx in class_to_id.items()}
        label_maps = {}

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            print(f"  -> Running SAM 3 PCS on {img_path.name} ({img.width}x{img.height})...")
            lmap = segmenter.predict_semantic(
                image=img,
                classes=classes,
                class_to_id=class_to_id,
                task_name=task_name,
                fill_gaps=fill_gaps,
            )
            label_maps[img_path.name] = lmap

            total_px = lmap.size
            cov_stats = {}
            for cid, cname in id_to_class.items():
                cnt = int((lmap == cid).sum())
                if cnt > 0:
                    cov_stats[cname] = round(100.0 * cnt / total_px, 2)
            unlabeled_pct = round(100.0 * int((lmap == 255).sum()) / total_px, 2)
            print(f"     Class coverage (%): {cov_stats} | Unlabeled: {unlabeled_pct}%")

            preview_path = vis_dir / task_name / f"{img_path.stem}_preview.jpg"
            save_visual_preview(
                image=img,
                output_path=preview_path,
                colors=colors,
                semantic_map=lmap,
                id_to_class=id_to_class,
            )
            task_summary["images"][img_path.name] = {
                "coverage_percent": cov_stats,
                "unlabeled_percent": unlabeled_pct,
                "preview": str(preview_path),
            }

        export_semantic_zip(
            output_zip=zip_path,
            label_maps=label_maps,
            classes=classes,
            colors=colors,
            class_to_id=class_to_id,
        )
        print(f"  [OK] Exported Segmentation mask 1.1 ZIP: {zip_path}")

    else:
        predictions_by_image = {}
        image_sizes = {}

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            image_sizes[img_path.name] = (img.height, img.width)
            print(f"  -> Running SAM 3 PCS on {img_path.name} ({img.width}x{img.height})...")

            if task_type == "panoptic":
                instances = segmenter.predict_panoptic(
                    image=img,
                    classes=classes,
                    stuff_classes=meta.get("stuff", []),
                    task_name=task_name,
                )
            else:
                instances = segmenter.predict_instances(
                    image=img,
                    classes=classes,
                    task_name=task_name,
                )

            predictions_by_image[img_path.name] = instances

            counts_by_cls: dict[str, int] = {}
            for inst in instances:
                counts_by_cls[inst.class_name] = counts_by_cls.get(inst.class_name, 0) + 1
            print(f"     Found {len(instances)} masks: {counts_by_cls}")

            preview_path = vis_dir / task_name / f"{img_path.stem}_preview.jpg"
            save_visual_preview(
                image=img,
                output_path=preview_path,
                colors=colors,
                instances=instances,
            )
            task_summary["images"][img_path.name] = {
                "total_instances": len(instances),
                "counts_by_class": counts_by_cls,
                "preview": str(preview_path),
            }

        export_coco_zip(
            output_zip=zip_path,
            predictions_by_image=predictions_by_image,
            image_sizes=image_sizes,
            classes=classes,
            segmentation_format=mask_format,  # type: ignore[arg-type]
        )
        print(f"  [OK] Exported COCO 1.0 ZIP ({mask_format}): {zip_path}")

    return task_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM 3 (Segment Anything Model 3) + CVAT 2D Image Segmentation Toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `predict-image`
    p_img = subparsers.add_parser("predict-image", help="Segment a single image with text prompts.")
    p_img.add_argument("--image", "-i", type=Path, required=True, help="Path to input image file.")
    p_img.add_argument("--prompts", "-p", nargs="+", required=True, help="One or more text concepts (e.g. car person road).")
    p_img.add_argument("--output", "-o", type=Path, default=Path("output_preview.jpg"), help="Path to save preview overlay image.")
    p_img.add_argument("--backend", default="auto", choices=["auto", "transformers", "ultralytics"])
    p_img.add_argument("--model-id", default="facebook/sam3", help="HF model id or checkpoint path.")
    p_img.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_img.add_argument("--threshold", type=float, default=0.35)

    # 2. `predict-folder`
    p_fld = subparsers.add_parser("predict-folder", help="Segment all images in a folder and export COCO 1.0 ZIP.")
    p_fld.add_argument("--input-dir", "-i", type=Path, required=True, help="Input directory of images.")
    p_fld.add_argument("--prompts", "-p", nargs="+", required=True, help="List of class names/prompts.")
    p_fld.add_argument("--output-zip", "-o", type=Path, default=Path("output_coco.zip"), help="Path to save COCO 1.0 ZIP.")
    p_fld.add_argument("--vis-dir", type=Path, default=Path("previews"), help="Directory to save visual overlays.")
    p_fld.add_argument("--backend", default="auto", choices=["auto", "transformers", "ultralytics"])
    p_fld.add_argument("--model-id", default="facebook/sam3")
    p_fld.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_fld.add_argument("--threshold", type=float, default=0.35)

    # Common manifest/dataset args
    def add_manifest_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--manifest",
            "-m",
            type=Path,
            default=Path("data/manifest.json"),
            help="Path to manifest.json (default: data/manifest.json).",
        )
        sub.add_argument(
            "--data-root",
            type=Path,
            default=Path("data"),
            help="Base directory containing task folders (default: data/).",
        )
        sub.add_argument(
            "--task",
            "-t",
            nargs="+",
            help="One or more task names to process.",
        )
        sub.add_argument(
            "--all",
            "-a",
            action="store_true",
            help="Run all tasks in the manifest.",
        )

    def add_cvat_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--host",
            default=os.environ.get("CVAT_HOST", "http://localhost:8080"),
            help="CVAT local host URL (default: http://localhost:8080).",
        )
        sub.add_argument(
            "--username",
            "-u",
            default=os.environ.get("CVAT_USERNAME", "admin"),
            help="CVAT username.",
        )
        sub.add_argument(
            "--password",
            "-p",
            default=os.environ.get("CVAT_PASSWORD", "admin"),
            help="CVAT password.",
        )

    # 3. `predict`
    p_pred = subparsers.add_parser("predict", help="Run SAM 3 on dataset tasks defined in manifest.json.")
    add_manifest_args(p_pred)
    p_pred.add_argument("--backend", default="auto", choices=["auto", "transformers", "ultralytics"])
    p_pred.add_argument("--model-id", default="facebook/sam3")
    p_pred.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_pred.add_argument("--threshold", type=float, default=0.35)
    p_pred.add_argument("--mask-threshold", type=float, default=0.50)
    p_pred.add_argument("--mask-format", choices=["polygon", "rle"], default="polygon")
    p_pred.add_argument("--fill-gaps", action="store_true")
    p_pred.add_argument("--out-dir", type=Path, default=Path("submissions"))
    p_pred.add_argument("--vis-dir", type=Path, default=Path("data/_cache/sam3_vis"))

    # 4. `cvat-push`
    p_push = subparsers.add_parser("cvat-push", help="Create tasks on CVAT and upload pre-annotations.")
    add_manifest_args(p_push)
    add_cvat_args(p_push)
    p_push.add_argument("--zip-dir", type=Path, default=Path("submissions"))
    p_push.add_argument("--recreate", action="store_true")
    p_push.add_argument("--skip-annotations", action="store_true")

    # 5. `cvat-pull`
    p_pull = subparsers.add_parser("cvat-pull", help="Export annotations from CVAT tasks into ZIPs.")
    add_manifest_args(p_pull)
    add_cvat_args(p_pull)
    p_pull.add_argument("--out-dir", type=Path, default=Path("submissions"))

    return parser


def main() -> int:
    from PIL import Image

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict-image":
        segmenter = Sam3Segmenter(
            model_id=args.model_id,
            backend=args.backend,
            device=args.device,
            default_threshold=args.threshold,
        )
        img = Image.open(args.image).convert("RGB")
        print(f"Segmenting {args.image} with concepts: {args.prompts}...")
        instances = segmenter.detect_classes(img, args.prompts)
        print(f"Found {len(instances)} instances.")
        save_visual_preview(img, args.output, colors={}, instances=instances)
        print(f"Saved visual overlay to: {args.output}")
        return 0

    elif args.command == "predict-folder":
        segmenter = Sam3Segmenter(
            model_id=args.model_id,
            backend=args.backend,
            device=args.device,
            default_threshold=args.threshold,
        )
        img_paths = sorted(
            list(args.input_dir.glob("*.jpg"))
            + list(args.input_dir.glob("*.png"))
            + list(args.input_dir.glob("*.jpeg"))
        )
        if not img_paths:
            print(f"No images found in {args.input_dir}")
            return 1

        preds_by_img = {}
        sizes = {}
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            sizes[p.name] = (img.height, img.width)
            print(f"Processing {p.name}...")
            insts = segmenter.detect_classes(img, args.prompts)
            preds_by_img[p.name] = insts
            save_visual_preview(img, args.vis_dir / f"{p.stem}_preview.jpg", colors={}, instances=insts)

        export_coco_zip(args.output_zip, preds_by_img, sizes, args.prompts, "polygon")
        print(f"Exported COCO 1.0 ZIP to: {args.output_zip}")
        return 0

    # Manifest-based commands
    manifest = load_manifest(args.manifest)
    selected_tasks = (
        list(manifest.keys())
        if (args.all or not args.task)
        else [t for t in args.task if t in manifest]
    )

    if args.command == "predict":
        segmenter = Sam3Segmenter(
            model_id=args.model_id,
            backend=args.backend,
            device=args.device,
            default_threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
        summaries = {}
        for tname in selected_tasks:
            meta = load_task_metadata(tname, manifest[tname], args.data_root)
            summaries[tname] = run_predict_for_task(
                segmenter=segmenter,
                meta=meta,
                out_dir=args.out_dir,
                vis_dir=args.vis_dir,
                mask_format=args.mask_format,
                fill_gaps=args.fill_gaps,
            )
        summary_file = args.vis_dir / "sam3_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[Summary] Statistics saved to: {summary_file}")
        return 0

    elif args.command == "cvat-push":
        cvat = CvatAutomationClient(
            host=args.host,
            username=args.username,
            password=args.password,
        )
        for tname in selected_tasks:
            meta = load_task_metadata(tname, manifest[tname], args.data_root)
            cvat.ensure_task(
                task_name=tname,
                labels_json_path=meta["cvat_labels_path"],
                image_paths=meta["images"],
                recreate=args.recreate,
            )
            if not args.skip_annotations:
                zip_path = args.zip_dir / f"{tname}.zip"
                if zip_path.is_file():
                    cvat.upload_preannotations(
                        task_name=tname,
                        task_type=meta["type"],
                        zip_path=zip_path,
                    )
        print("\n[Done] All tasks pushed to CVAT. Open CVAT to review and edit annotations!")
        return 0

    elif args.command == "cvat-pull":
        cvat = CvatAutomationClient(
            host=args.host,
            username=args.username,
            password=args.password,
        )
        for tname in selected_tasks:
            meta = load_task_metadata(tname, manifest[tname], args.data_root)
            out_zip = args.out_dir / f"{tname}.zip"
            cvat.export_task_dataset(
                task_name=tname,
                task_type=meta["type"],
                output_zip=out_zip,
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
