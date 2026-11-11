"""CVAT Local Automation Client using `cvat-sdk`.

Automates the repetitive CVAT steps for all 9 Day 3 tasks:
  1. Creating each task with its exact `cvat-labels.json` schema and uploading its JPG images.
  2. Importing SAM 3 pre-annotation ZIPs into CVAT tasks.
     (Note: For semantic tasks, CVAT's `Segmentation mask 1.1` importer creates a `'background'`
      polygon for `(0,0,0)` pixels which fails if `'background'` is not in `cvat-labels.json`.
      Therefore, `upload_preannotations` automatically converts `Segmentation mask 1.1` into
      `COCO 1.0` polygons of only the registered task classes when importing into CVAT, while
      `export_task_dataset` exports semantic tasks back out as `Segmentation mask 1.1`!)
  3. Exporting finished/edited annotations from CVAT back into `submissions/<task>.zip`.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator


def get_cvat_format_name(task_type: str) -> str:
    """Return exact CVAT dataset/annotation format name for exporting a Day 3 task type."""
    if task_type == "semantic":
        return "Segmentation mask 1.1"
    return "COCO 1.0"


def convert_semantic_zip_to_coco_for_cvat_import(
    semantic_zip_path: Path,
    output_coco_zip: Path,
) -> Path:
    """Convert a `Segmentation mask 1.1` ZIP into a `COCO 1.0` polygon ZIP for CVAT import.

    Excludes `'background'` / `(0, 0, 0)` so CVAT never raises:
    `CvatImportError: Label 'background' is not registered for this task`.
    """
    import numpy as np
    from PIL import Image
    try:
        from .cvat_exporter import export_coco_zip
        from .sam3_engine import PredictedInstance
    except ImportError:
        from cvat_exporter import export_coco_zip
        from sam3_engine import PredictedInstance

    with zipfile.ZipFile(semantic_zip_path, "r") as zf:
        names = zf.namelist()
        lm_key = next((k for k in names if k.endswith("labelmap.txt")), None)
        if not lm_key:
            raise ValueError(f"Không tìm thấy labelmap.txt trong {semantic_zip_path}")

        lm_text = zf.read(lm_key).decode("utf-8", errors="ignore")
        classes: list[str] = []
        color_to_cls: dict[tuple[int, int, int], str] = {}

        for line in lm_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            cls_name, rgb_str = line.split(":", 1)[0].strip(), line.split(":", 2)[1].strip()
            if cls_name in ("background", "void", "unlabeled") or not rgb_str:
                continue
            r, g, b = (int(x) for x in rgb_str.split(","))
            if cls_name not in classes:
                classes.append(cls_name)
            color_to_cls[(r, g, b)] = cls_name

        predictions_by_image: dict[str, list[PredictedInstance]] = {}
        image_sizes: dict[str, tuple[int, int]] = {}

        for key in sorted(names):
            if "SegmentationClass/" not in key or not key.lower().endswith(".png"):
                continue
            stem = Path(key).stem
            jpg_name = f"{stem}.jpg"
            rgb = np.array(Image.open(io.BytesIO(zf.read(key))).convert("RGB"))
            h, w = rgb.shape[:2]
            image_sizes[jpg_name] = (h, w)
            inst_list: list[PredictedInstance] = []

            for (r, g, b), cls_name in color_to_cls.items():
                mask = (rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)
                if int(mask.sum()) >= 12:
                    inst_list.append(
                        PredictedInstance(
                            class_name=cls_name,
                            mask=mask,
                            score=1.0,
                        )
                    )
            predictions_by_image[jpg_name] = inst_list

    return export_coco_zip(
        output_zip=output_coco_zip,
        predictions_by_image=predictions_by_image,
        image_sizes=image_sizes,
        classes=classes,
        segmentation_format="polygon",
    )


class CvatAutomationClient:
    """High-level wrapper around `cvat_sdk` for managing Day 3 tasks on local CVAT."""

    def __init__(
        self,
        host: str = "http://localhost:8080",
        username: str = "admin",
        password: str = "admin",
    ) -> None:
        self.host = host.rstrip("/")
        self.username = username
        self.password = password

    @contextlib.contextmanager
    def _connect(self) -> Iterator[Any]:
        try:
            from cvat_sdk.core.client import Client
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài đặt `cvat-sdk`. Hãy chạy: pip install cvat-sdk"
            ) from exc

        client = Client(self.host, check_server_version=False)
        client.login((self.username, self.password))
        try:
            yield client
        finally:
            client.logout()

    def find_task_by_name(self, client: Any, task_name: str) -> Any | None:
        """Find the most recent CVAT task matching `task_name` exactly."""
        tasks = client.tasks.list()
        matches = [t for t in tasks if getattr(t, "name", "") == task_name]
        if not matches:
            return None
        matches.sort(key=lambda t: getattr(t, "id", 0), reverse=True)
        return matches[0]

    def ensure_task(
        self,
        task_name: str,
        labels_json_path: Path,
        image_paths: list[Path],
        recreate: bool = False,
    ) -> int:
        """Create a CVAT task with its `cvat-labels.json` and upload its images if not existing."""
        from cvat_sdk.core.proxies.tasks import ResourceType

        labels = json.loads(labels_json_path.read_text(encoding="utf-8"))
        sorted_images = [str(p.resolve()) for p in sorted(image_paths)]

        with self._connect() as client:
            existing = self.find_task_by_name(client, task_name)
            if existing is not None:
                if recreate:
                    print(f"[CVAT] Xóa task cũ '{task_name}' (id={existing.id}) để tạo lại...")
                    existing.remove()
                else:
                    print(f"[CVAT] Task '{task_name}' đã tồn tại (id={existing.id}), giữ nguyên.")
                    return int(existing.id)

            print(
                f"[CVAT] Đang tạo task '{task_name}' với {len(labels)} labels và {len(sorted_images)} ảnh..."
            )
            task_spec = {
                "name": task_name,
                "labels": labels,
            }
            task = client.tasks.create_from_data(
                spec=task_spec,
                resource_type=ResourceType.LOCAL,
                resources=sorted_images,
                data_params={"image_quality": 95},
            )
            print(f"[CVAT] Đã tạo xong task '{task_name}' -> Task ID: {task.id}")
            return int(task.id)

    def upload_preannotations(
        self,
        task_name: str,
        task_type: str,
        zip_path: Path,
    ) -> None:
        """Upload a SAM 3 generated annotation ZIP to CVAT.

        For semantic tasks (`Segmentation mask 1.1`), converts the PNG masks into
        temporary `COCO 1.0` polygons of only registered task classes so CVAT does
        not fail on `'background'` pixels.
        """
        if not zip_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file pre-annotation ZIP: {zip_path}")

        with self._connect() as client:
            task = self.find_task_by_name(client, task_name)
            if task is None:
                raise ValueError(
                    f"Không tìm thấy task '{task_name}' trên CVAT ({self.host}). "
                    f"Hãy chạy bước tạo task trước."
                )

            if task_type == "semantic":
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_coco = Path(tmp_dir) / f"{task_name}_cvat_import.zip"
                    convert_semantic_zip_to_coco_for_cvat_import(zip_path, tmp_coco)
                    print(
                        f"[CVAT] Đang đẩy pre-annotation '{zip_path.name}' (chuyển sang COCO 1.0 polygons để tránh lỗi label 'background') "
                        f"lên task '{task_name}' (id={task.id})..."
                    )
                    task.import_annotations("COCO 1.0", str(tmp_coco.resolve()))
            else:
                format_name = get_cvat_format_name(task_type)
                print(
                    f"[CVAT] Đang đẩy pre-annotation '{zip_path.name}' ({format_name}) "
                    f"lên task '{task_name}' (id={task.id})..."
                )
                task.import_annotations(format_name, str(zip_path.resolve()))

            print(f"[CVAT] Đã import thành công pre-annotations cho '{task_name}'!")

    def list_existing_tasks(self) -> list[tuple[int, str]]:
        """Return list of `(task_id, task_name)` currently on CVAT."""
        with self._connect() as client:
            tasks = client.tasks.list()
            return [(int(getattr(t, "id", 0)), str(getattr(t, "name", ""))) for t in tasks]

    def export_task_dataset(
        self,
        task_name: str,
        task_type: str,
        output_zip: Path,
    ) -> Path | None:
        """Export annotations from a CVAT task into `submissions/<task_name>.zip`."""
        format_name = get_cvat_format_name(task_type)
        output_zip.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as client:
            task = self.find_task_by_name(client, task_name)
            if task is None:
                print(
                    f"[CVAT] Bỏ qua '{task_name}': Chưa có task này trên CVAT ({self.host}). "
                    f"Hãy chạy `cvat-push` trước để tạo task và đẩy pre-annotation lên CVAT."
                )
                return None

            print(
                f"[CVAT] Đang export task '{task_name}' (id={task.id}) "
                f"theo định dạng '{format_name}' -> {output_zip}..."
            )
            if output_zip.exists():
                output_zip.unlink()
            task.export_dataset(
                format_name=format_name,
                filename=str(output_zip.resolve()),
                include_images=False,
            )
            time.sleep(0.2)
            print(f"[CVAT] Đã xuất thành công: {output_zip}")
            return output_zip
