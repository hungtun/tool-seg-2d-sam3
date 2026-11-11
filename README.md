# tool-seg-2d-sam3: SAM 3 & CVAT 2D Image Segmentation Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS & CUDA](https://img.shields.io/badge/PyTorch-Apple%20MPS%20%7C%20CUDA-green.svg)](https://pytorch.org/)
[![SAM 3](https://img.shields.io/badge/Model-Meta%20SAM%203-red.svg)](https://huggingface.co/facebook/sam3)
[![CVAT SDK](https://img.shields.io/badge/Integration-CVAT%20SDK-orange.svg)](https://github.com/cvat-ai/cvat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end automated 2D image segmentation toolkit integrating **Meta's Segment Anything Model 3 (SAM 3)** with the **Computer Vision Annotation Tool (CVAT)**.

Designed for high-throughput AI pre-annotation, street-scene segmentation, and seamless synchronization with local or remote CVAT servers.

---

## 🌟 Key Features

1. **Promptable Concept Segmentation (PCS) with SAM 3**:
   - Detects and segments all object instances from open-vocabulary text phrases (e.g. `"car"`, `"pedestrian"`, `"traffic sign"`, `"building"`) without manual click prompts.
   - Dual-backend support: Hugging Face `transformers` (`facebook/sam3`, `tkclam/sam3`) and `ultralytics` (`sam3.pt`).
2. **Optimized for Apple Silicon (`mps`) & CUDA**:
   - Native hardware acceleration on macOS Apple Silicon (M1 / M2 / M3 / M4) via PyTorch `mps` with automatic fallback.
3. **Multi-Paradigm Segmentation & Task Post-Processing**:
   - **Semantic Segmentation**: Produces `Segmentation mask 1.1` (`labelmap.txt` + `SegmentationClass/*.png`), resolving stuff layers by confidence score and painting thin foreground objects last.
   - **Instance Segmentation**: Exports `COCO 1.0` polygons with Mask NMS deduplication, hole filling inside contours, and boundary overlap resolution.
   - **Panoptic Segmentation**: Merges stuff classes while isolating separate thing instances, matching standard Panoptic Quality (PQ) contracts.
4. **Full CVAT Automation via `cvat-sdk`**:
   - **`cvat-push`**: Automatically creates tasks on CVAT, loads custom `cvat-labels.json`, uploads batch images, and injects SAM 3 pre-annotations.
   - **`cvat-pull`**: Automatically exports finished datasets from CVAT into standard ZIP packages.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A["Raw Images + Taxonomy (classes.json)"] --> B["Sam3Segmenter (SAM 3 Engine)"]
    B -->|MPS / CUDA Inference| C{"Post-Processing Module"}
    C -->|Semantic Engine| D["Segmentation mask 1.1 (PNGs + labelmap.txt)"]
    C -->|Instance Engine (Mask NMS + Hole Filling)| E["COCO 1.0 Polygons (instances_default.json)"]
    C -->|Panoptic Engine (Stuff Merge + Thing Separation)| F["COCO 1.0 Panoptic"]
    D & E & F --> G["CvatAutomationClient (cvat-sdk)"]
    G -->|cvat-push| H["CVAT Local Server (Review & Refine)"]
    H -->|cvat-pull| I["Final Evaluated & Annotated Datasets"]
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and run the setup script:

```bash
git clone https://github.com/hungtun/tool-seg-2d-sam3.git
cd tool-seg-2d-sam3

# Automated setup (creates .venv with PyTorch MPS / CUDA):
bash setup_env.sh
source .venv/bin/activate
```

*(Optional)* If accessing gated weights from `facebook/sam3`, export your Hugging Face token (the tool also automatically falls back to ungated mirrors like `tkclam/sam3`):
```bash
export HF_TOKEN="hf_your_token_here"
```

---

## 💻 CLI Usage

### A. Segment a Single Image
Run concept segmentation on an image with text prompts:
```bash
python cli.py predict-image \
  --image sample.jpg \
  --prompts "car" "person" "bicycle" \
  --output result_overlay.jpg
```

### B. Segment an Entire Directory
Segment all images in a folder and export to a CVAT `COCO 1.0` ZIP:
```bash
python cli.py predict-folder \
  --input-dir path/to/images/ \
  --prompts "car" "truck" "traffic light" \
  --output-zip annotations_coco.zip
```

### C. Run on Dataset Tasks from `manifest.json`
Run SAM 3 automated segmentation across all tasks in a dataset:
```bash
python cli.py predict \
  --manifest data/manifest.json \
  --all \
  --threshold 0.35
```

### D. Synchronize with CVAT Local

**1. Create tasks & push pre-annotations to CVAT:**
```bash
python cli.py cvat-push \
  --manifest data/manifest.json \
  --all \
  --host http://localhost:8080 \
  --username admin \
  --password admin
```

**2. Review, refine boundaries, and click Save inside CVAT UI.**

**3. Pull finished annotations back to disk:**
```bash
python cli.py cvat-pull \
  --manifest data/manifest.json \
  --all \
  --host http://localhost:8080 \
  --username admin \
  --password admin
```

---

## 📦 Python API Usage

You can also use the toolkit directly inside your Python scripts or Jupyter notebooks:

```python
from PIL import Image
from sam3_cvat import Sam3Segmenter, export_coco_zip, save_visual_preview

# Initialize segmenter (auto-detects Apple MPS / CUDA)
segmenter = Sam3Segmenter(model_id="facebook/sam3", default_threshold=0.35)

# Load image
image = Image.open("street.jpg").convert("RGB")

# Detect multiple concepts simultaneously
instances = segmenter.detect_classes(image, classes=["road", "car", "person"])
print(f"Found {len(instances)} instances.")

# Save visual overlay
save_visual_preview(image, "preview.jpg", colors={}, instances=instances)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Model weights are subject to Meta's SAM License.
