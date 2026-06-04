# Retail Shelf Monitoring

Retail Shelf Monitoring is a computer vision project that uses a YOLOv8 object detection model to detect retail shelf products, estimate shelf occupancy, and generate restock recommendations from shelf images.

This project was built as a deep learning computer vision application using the SKU110K retail shelf dataset.

---

## Project Overview

Retail stores often rely on manual shelf checks to identify low-stock shelves, misplaced items, and stockout risks. This process can be time-consuming and inconsistent.

This project demonstrates an AI-powered shelf monitoring prototype that:

- Detects visible products in retail shelf images
- Counts detected product instances
- Estimates shelf occupancy using detected bounding-box coverage
- Identifies empty shelf space
- Generates inventory status recommendations
- Provides a Streamlit dashboard for interactive use

---

## Repository Structure

| Folder/File | Purpose |
|---|---|
| `app/streamlit_app.py` | Streamlit dashboard for shelf monitoring demo |
| `preprocessing/data.yaml` | YOLO dataset configuration |
| `preprocessing/train.py` | YOLOv8 training script |
| `preprocessing/dataset prep.ipynb` | Dataset sanity check and bounding-box visualization |
| `src/predict.py` | Inference script for single shelf image analysis |
| `src/evaluate_test.py` | Independent test set evaluation script |
| `test_images/` | Sample images used for demo |
| `.gitignore` | Prevents dataset, model weights, venv, and generated outputs from being pushed |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Team & Roles

| Team Member | Role | Responsibilities |
|---|---|---|
| Om | Data Lead | Dataset preparation, SKU110K formatting, train/validation/test structure |
| Vishal | Model Lead | YOLOv8 training, evaluation, inference pipeline |
| Aditya | Analytics Lead | Product count, shelf occupancy, zone-wise analytics, restock logic |
| Shweta | App & Deployment Lead | Streamlit dashboard, local deployment, demo workflow |

---

## Problem Statement

Retail shelf monitoring is still largely manual in many store environments. Employees must physically inspect shelves to identify:

- Low inventory
- Empty shelf regions
- Stockout risk
- Poor shelf visibility
- Areas needing restocking

Manual checks are slow, inconsistent, and difficult to scale across stores.

---

## Proposed Solution

The system takes a shelf image as input and uses a YOLOv8 object detection model to detect product instances. These detections are then passed into an analytics layer that calculates shelf-level business metrics.

### System Flow

```text
Retail Shelf Image
        ↓
YOLOv8 Object Detection
        ↓
Product Bounding Boxes
        ↓
Analytics Layer
  - Product count
  - Average confidence
  - Shelf occupancy
  - Empty space estimate
  - Zone-wise product count
        ↓
Business Rule Engine
  - Restock Required
  - Monitor Shelf
  - Healthy Shelf
        ↓
Streamlit Dashboard

```

---

## Dataset

This project uses a subset of the SKU110K dataset.

SKU110K is a dense retail shelf object detection dataset containing shelf images and bounding-box annotations for product instances.

The dataset uses a single class:

0: product

---

## Dataset Structure

The expected local dataset structure is:

```text

preprocessing/
└── dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

```

---

## Dataset Split

| Split | Images | Purpose |
|---|---:|---|
| Train | ~8,192 | Model learning |
| Validation | 584 | Monitoring during training |
| Test | 2,920 | Independent final evaluation |

The validation set contained 90,456 labeled product instances, and the test set contained 429,411 labeled product instances.

---

## Model

The project uses YOLOv8n from Ultralytics.


| Component | Details |
|---|---|
| Model | YOLOv8n |
| Framework | PyTorch / Ultralytics |
| Task | Object Detection |
| Class | Product |
| Training Hardware | NVIDIA RTX 3070 Laptop GPU |
| Epochs | 10 |
| Image Size | 640 |
| Batch Size | 8 |

---

## Validation Results

| Metric | Score |
|---|---:|
| Precision | 88.7% |
| Recall | 80.5% |
| mAP50 | 85.6% |
| mAP50-95 | 51.2% |

---

## Independent Test Results

| Metric | Score |
|---|---:|
| Precision | 88.8% |
| Recall | 82.1% |
| mAP50 | 88.2% |
| mAP50-95 | 52.8% |

---

## Demo Example: Ice Cream Shelf

| Metric | Value |
|---|---:|
| Products Detected | 10 |
| Average Confidence | 0.64 |
| Shelf Occupancy | 32.71% |
| Estimated Empty Space | 67.29% |
| Inventory Status | Restock Required |

---

## Demo Example: Vegetable Shelf

| Metric | Value |
|---|---:|
| Products Detected | 13 |
| Average Confidence | 0.61 |
| Shelf Occupancy | 40.80% |
| Estimated Empty Space | 59.20% |
| Inventory Status | Monitor Shelf |

---

## Business Rule Logic

| Shelf Occupancy | Status |
|---:|---|
| < 40% | Restock Required |
| 40% – 70% | Monitor Shelf |
| > 70% | Healthy Shelf |

---

## Limitations

- The current model detects generic products, not exact SKU or brand identities.
- SKU110K uses one product class, so the system does not classify individual brands.
- Shelf occupancy is estimated using bounding-box area, not true physical shelf capacity.
- Dense overlapping products may lead to missed detections.
- Reflections, blurry images, poor lighting, and unusual camera angles can reduce performance.
- The current prototype uses uploaded images, not live camera streams.

---

## Future Improvements

- Add SKU-level product classification.
- Add CLIP-based product similarity search.
- Add multi-image batch upload for store-wide shelf audits.
- Add live camera integration.
- Store historical shelf monitoring data in a database.
- Add alerting through email, Slack, or store operations tools.
- Deploy a production version using React, FastAPI, and cloud GPU inference.

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- PIL
- SKU110K Dataset

---

## Summary

This project demonstrates an end-to-end computer vision application for retail shelf monitoring. It combines YOLOv8 object detection with a business analytics layer to convert shelf images into actionable inventory insights.
