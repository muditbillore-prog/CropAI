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

The project uses a trained YOLO classification model covering 33 rice variety classes.

## Development Status

CropAI is under active development. This repository will document the project's implementation, model integration, evaluation and improvements progressively.

## Note

Model confidence scores are predictions, not guarantees of real-world accuracy. Performance can vary with image quality, lighting, camera angle and field conditions.