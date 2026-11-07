# CropAI — AI-Based Crop Identification and Farm Analysis

CropAI is an AI-powered application developed to explore rice variety identification using image classification and video analysis.

The project combines computer vision, deep learning and a Streamlit-based interface to process uploaded farm images and videos.

## Project Overview

The application is designed to:
- Identify rice varieties from uploaded images.
- Analyse uploaded farm videos using sampled frames.
- Generate prediction summaries and downloadable CSV reports.
- Display model confidence scores for predicted rice varieties.

## Technology Stack

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- Pandas
- Pillow

## Current Model

The project uses a YOLO11n classification model (`models/best.pt`, about 3.2 MB) covering 33 rice variety classes.

## Run Locally

Developed and tested on Windows 11 with Python 3.12.10. The trained model is included in the repository, so no training is needed to run the app.

    git clone https://github.com/muditbillore-prog/CropAI.git
    cd CropAI
    python -m venv .venv
    source .venv/Scripts/activate      # Git Bash on Windows
    pip install -r requirements.txt
    streamlit run app.py

On Linux or macOS, activate the environment with `source .venv/bin/activate` instead. The training dataset is not included in this repository.

## Evaluation Results

The model was evaluated on a held-out test split of 324 images across 33 classes that were not used for training or validation.

| Metric | Result |
| --- | --- |
| Independent test Top-1 accuracy | 99.07% |
| Independent test Top-5 accuracy | 100.00% |
| Test images | 324 |
| Classes | 33 |

Validation accuracy (99.669% Top-1) is a separate measurement taken during training and is not the test result.

![Normalized confusion matrix on the independent test set](docs/images/confusion_matrix_test.png)

Full methodology, dataset details and training configuration are in [docs/MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md). The test evaluation can be reproduced with `evaluate_rice_classifier.py`, provided the dataset is available locally.

## Limitations

- CropAI classifies the rice variety shown in an image. It does not detect individual plants.
- It does not diagnose crop disease or provide a validated crop-health score.
- Video analysis works on sampled frames and is not a field-level survey.
- All reported accuracy comes from a held-out split of a single dataset. Performance on other fields, cameras, lighting or growth stages has not been measured.
- Model confidence scores are predictions, not guarantees of real-world accuracy.

## Development Status

CropAI is under active development. This repository documents the project's implementation, model integration, evaluation and improvements progressively.
