# AnnotationFlow

![License: MIT](https://img.shields.io/badge/license-MIT-97d700?labelColor=555555&style=flat)
![Release: v0.1.0](https://img.shields.io/badge/release-v0.1.0-ff7a2f?labelColor=555555&style=flat)

AnnotationFlow is a local software tool for building YOLO object-detection datasets from raw image folders. It normalizes image files, removes exact duplicates, runs images through a configurable Roboflow Workflow, converts detections into YOLO labels, splits the result into `train`/`valid`/`test`, and exports a ready-to-train dataset ZIP.

<img src="docs/demo.png?raw=1" alt="AnnotationFlow demo" />

## What It Does

AnnotationFlow turns image batches into clean object-detection datasets:

```text
raw images
  -> image validation and JPG normalization
  -> exact duplicate detection
  -> Roboflow Workflow inference
  -> detection extraction
  -> YOLO label writing
  -> train / valid / test split
  -> downloadable dataset ZIP
```

The application includes a FastAPI backend and a React/Vite frontend. Jobs run in the background, while the UI shows live command-style logs and a four-step progress roadmap.

<h2>
  <img src="https://app.roboflow.com/images/wordmark-purboflow.svg" alt="Roboflow" height="28" style="vertical-align: middle; margin-right: 10px;" />
  <span style="vertical-align: middle;">Roboflow Workflow Integration</span>
</h2>

AnnotationFlow uses Roboflow Workflows as the inference layer. This keeps dataset generation flexible: you can update the model, workflow graph, confidence behavior, and output structure in Roboflow while keeping the local export pipeline stable.

The UI can optionally override these values per run:

```env
ROBOFLOW_API_KEY
ROBOFLOW_WORKSPACE_NAME
ROBOFLOW_WORKFLOW_ID
ROBOFLOW_USE_CACHE
ROBOFLOW_CONFIDENCE
TRAIN_RATIO
VAL_RATIO
TEST_RATIO
```

If a field is left empty in the UI, the backend falls back to `.env`. This allows repeatable default settings while still supporting one-off experiments from the browser.

<h2>
  <img src="https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/6914c5f6e5b3ebb12ce86156_updated-logo.svg" alt="Ultralytics" height="28" style="vertical-align: middle; margin-right: 10px;" />
  <span style="vertical-align: middle;">YOLO Object-Detection Format</span>
</h2>

For every processed image, AnnotationFlow writes a matching `.txt` label file. Each detection is exported as:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `0..1`:

```text
x_center = prediction.x / image.width
y_center = prediction.y / image.height
width    = prediction.width / image.width
height   = prediction.height / image.height
```

The exported dataset has the standard YOLO layout:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Example `data.yaml`:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['stair']
```

## Requirements

- Python `>=3.12,<3.13`
- Node.js compatible with Vite `7.x`
- `uv` for backend dependency management
- npm for frontend dependency management
- A Roboflow API key with access to the selected workspace and workflow

Backend runtime dependencies:

- FastAPI
- Uvicorn
- python-multipart
- pydantic-settings
- Pillow
- pillow-heif
- inference-sdk

Frontend runtime dependencies:

- React
- React DOM
- Vite

## Configuration

Create a local `.env` file:

```bash
cp .env.example .env
```

Required Roboflow settings:

```env
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_API_KEY=replace_with_your_roboflow_api_key
ROBOFLOW_WORKSPACE_NAME=your_workspace
ROBOFLOW_WORKFLOW_ID=your_workflow
ROBOFLOW_USE_CACHE=true
ROBOFLOW_CONFIDENCE=0.4
```

Dataset split settings:

```env
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SPLIT_SEED=42
```

The split ratios must add up to `1.0`.

## Local Development

Install dependencies:

```bash
./setup.sh
```

Run the full application:

```bash
./run.sh
```

Open:

```text
Frontend: http://127.0.0.1:8081
Backend:  http://127.0.0.1:8000
```

Run backend tests:

```bash
cd backend
uv run pytest
```

Run frontend checks:

```bash
cd frontend
npm test
```

## Contributing

Contributions are welcome. Good first areas include:

- Additional Roboflow Workflow response parsers
- Better validation and reporting for failed images
- Dataset export options
- UI improvements for large image batches
- More tests around edge cases and workflow outputs

Before opening a pull request, run:

```bash
cd backend && uv run pytest
cd ../frontend && npm test
```

Please keep API keys, private datasets, and generated runtime output out of source control.
