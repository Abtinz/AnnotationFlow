# AnnotationFlow

AnnotationFlow is an automated YOLO object-detection dataset builder. It cleans raw images, removes duplicates, converts image formats, sends each image through a user-defined Roboflow Workflow, extracts detections, converts them into YOLO labels, and exports a ready-to-train `train`/`valid`/`test` dataset.

This first phase targets YOLO object detection only. Segmentation, classification, active learning, and training orchestration are intentionally out of scope for the initial build.

## Workflow

```text
Raw images
  -> format validation
  -> JPG normalization
  -> exact duplicate removal
  -> Roboflow Workflow inference
  -> detection extraction
  -> YOLO label conversion
  -> train/valid/test split
  -> data.yaml export
```

## Project Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── jobs.py
│   │   ├── image_normalizer.py
│   │   ├── duplicate_detector.py
│   │   ├── roboflow_client.py
│   │   ├── yolo_writer.py
│   │   └── dataset_splitter.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and provide your Roboflow credentials. Do not commit `.env`.

```bash
cp .env.example .env
```

Required Roboflow settings:

```env
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_API_KEY=replace_with_your_roboflow_api_key
ROBOFLOW_WORKSPACE_NAME=abtinzandi
ROBOFLOW_WORKFLOW_ID=find-bus
ROBOFLOW_CONFIDENCE=0.4
```

The API key must be created in Roboflow and must have access to the configured workspace and workflow. If a key is exposed in chat, logs, or source control, rotate it before using the system again.

## Backend API

Planned first-phase endpoints:

```text
GET  /health
POST /jobs
GET  /jobs/{job_id}
GET  /jobs/{job_id}/logs
GET  /jobs/{job_id}/dataset
```

`POST /jobs` accepts multiple image files, creates a processing job, and returns a `job_id`. The frontend polls the job and log endpoints until processing is complete.

## YOLO Object-Detection Format

For each image, AnnotationFlow writes a matching `.txt` file. Each detection is one row:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `0..1` using the Roboflow image width and height:

```text
x_center = prediction.x / image.width
y_center = prediction.y / image.height
width = prediction.width / image.width
height = prediction.height / image.height
```

The exported dataset uses this shape:

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
names: ['bus']
```

## Local Development

Install backend dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run backend tests:

```bash
cd backend
pytest
```

Run the backend:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the frontend app with Vite:

```bash
cd frontend
npm run dev
```

## Docker

After creating `.env`, start the stack:

```bash
docker compose up --build
```

The backend listens on `http://localhost:8000`. The frontend dev server listens on `http://localhost:8081`.

## Runtime Output

Jobs write runtime data under `output/jobs/<job_id>`:

```text
output/jobs/<job_id>/
├── originals/
├── normalized/
├── roboflow/
│   ├── raw_results.jsonl
│   └── visualized/
├── dataset/
│   ├── train/images/
│   ├── train/labels/
│   ├── valid/images/
│   ├── valid/labels/
│   ├── test/images/
│   ├── test/labels/
│   └── data.yaml
├── logs.jsonl
└── summary.json
```

`output/`, `uploads/`, `.env`, and sample local image folders are ignored by git.
