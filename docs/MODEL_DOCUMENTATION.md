# CropAI — Model Documentation

## 1. Project Overview

CropAI is an AI-powered rice variety classification project designed to identify rice varieties from uploaded images and video frames.

The application uses a trained YOLO11 classification model to predict the most likely rice variety and display prediction confidence scores.

CropAI provides a Streamlit-based dashboard for image and video analysis, Top-5 predictions, CSV report generation, and maturity planning calculations.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning Framework | Ultralytics |
| Model Architecture | YOLO11n Classification |
| Image Processing | OpenCV |
| Dashboard | Streamlit |
| Data Handling | Pandas |
| Image Handling | Pillow |

## 3. Dataset Information

The project uses a cleaned rice classification dataset.

Dataset directory:

`datasets/rice_classification_clean_v1`

Dataset statistics:

| Dataset Split | Number of Images |
|---|---:|
| Training | 2,476 |
| Validation | 302 |
| Testing | 324 |
| Total | 3,102 |

The dataset contains 33 rice variety classes organized into separate class folders.

The dataset is not included in the GitHub repository.

A second split of the same images, `datasets/rice_classification_time_split_v2`, was created by `make_time_ordered_split.py` and is described in Section 7:

| Dataset Split | Number of Images |
|---|---:|
| Training | 2,148 |
| Validation | 270 |
| Testing | 288 |
| Dropped (buffer between blocks) | 396 |
| Total | 3,102 |

## 4. Model Architecture

CropAI uses the YOLO11n Classification model provided by Ultralytics.

The model is trained for image classification rather than object detection.

The model predicts the rice variety represented in an input image and returns class probabilities.

Trained model location:

`models/best.pt`

Model type: YOLO11n Classification

## 5. Training Configuration

The training script is:

`train_rice_classifier.py`

The configured training parameters are:

| Parameter | Value |
|---|---|
| Model | YOLO11n Classification |
| Epochs | 20 |
| Image Size | 224 × 224 |
| Batch Size | 8 |
| Device | CPU |
| Workers | 0 |
| Patience | 5 |
| Random Seed | 42 |

The training script checks the dataset directories and class-folder consistency before starting training.

It also prevents accidental overwriting of an existing training run.

## 6. Validation Results

The recorded validation results are:

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 99.669% |
| Top-5 Accuracy | 100% |

These results represent validation performance on the original random split and are reported separately from test-set performance. They share the overlap with the training images described in Section 7, so they overstate performance on new photos. A model trained on the time-ordered split reached 87.0% Top-1 and 93.7% Top-5 on its own validation set (270 images).

## 7. Test Evaluation

The original test evaluation (random split) was completed using:

`evaluate_rice_classifier.py`

The evaluation uses the separate test dataset split containing 324 images across 33 rice variety classes.

### Test Results (Original Random Split)

| Metric | Result |
|---|---:|
| Test Images | 324 |
| Number of Classes | 33 |
| Top-1 Accuracy | 99.07% |
| Top-5 Accuracy | 100.00% |

The model achieved 99.07% Top-1 accuracy and 100% Top-5 accuracy on this test split. These figures overstate performance on new photos because many test images have near-identical training images; see the overlap analysis below.

Top-1 accuracy measures whether the highest-ranked prediction matches the actual class.

Top-5 accuracy measures whether the correct class appears among the five highest-ranked predictions.

### Evaluation Configuration

| Parameter | Value |
|---|---|
| Dataset Split | Test |
| Image Size | 224 × 224 |
| Batch Size | 8 |
| Device | CPU |
| Workers | 0 |
| Framework | Ultralytics 8.4.183 |

### Confusion Matrix

The normalized confusion matrix provides a class-wise view of the model's classification performance.

![CropAI normalized test confusion matrix, original random split](images/confusion_matrix_test.png)

The diagonal represents correct classifications, while off-diagonal entries indicate misclassifications.

The evaluation results are specific to this test dataset and do not guarantee identical performance on unseen real-world field conditions.

### Overlap Between Training and Test Images

Each variety was photographed in one short sitting (file-name timestamps show one date per variety and mostly 1-6 minutes per variety), and the original split was random. For every test image, `check_split_leakage.py` finds the most similar training image of the same variety using a 64-bit difference hash (a crude similarity measure) and reports accuracy by Hamming distance:

| Distance to nearest training image | Test images | Correct | Accuracy |
|---|---:|---:|---:|
| 0-5 (near-identical) | 193 | 193 | 100.00% |
| 6-10 | 70 | 70 | 100.00% |
| 11-15 | 41 | 41 | 100.00% |
| > 15 (distinct) | 20 | 17 | 85.00% |

Many source files carry suffixes such as `_01`, `_1` and `_2` and share a capture time with another file; the origin of these files is unknown. Exact duplicates were removed by `clean_rice_dataset.py`, but near-copies were not. The 99.07% result therefore overstates how well the model handles photos that do not resemble its training images.

### Time-Ordered Split Evaluation

`make_time_ordered_split.py` builds a second dataset, `datasets/rice_classification_time_split_v2`. For each variety it sorts the images by the capture time in the file name and cuts them into consecutive train, validation and test blocks, dropping a few images (a buffer of 6) between blocks and cutting only between different seconds. The source dataset is not modified.

| Property | Random split (original) | Time-ordered split |
|---|---:|---:|
| Training / validation / test images | 2,476 / 302 / 324 | 2,148 / 270 / 288 |
| Test images with a near-identical training image (distance <= 5) | 193 | 34 |
| Test images distinct from training (distance > 15) | 20 | 233 |

A new model was trained on the stricter split with the same settings (YOLO11n-cls, 224 x 224, batch 8, CPU, seed 42, up to 20 epochs with early stopping). It stopped after 17 epochs, with the best epoch at 12 (about 15 minutes of training).

| Metric | Validation (270 images) | Test (288 images) |
|---|---:|---:|
| Top-1 Accuracy | 87.0% | 83.33% (240 / 288) |
| Top-5 Accuracy | 93.7% | 94.44% (272 / 288) |

![CropAI normalized confusion matrix, time-ordered test set](images/confusion_matrix_test_time_split.png)

This is the more realistic estimate, but it still measures photos taken in the same sittings as the training photos: 34 test images still have a near-identical training image, and the results say nothing about other days, fields, cameras or lighting. With 288 test images the 95% interval is about +/-4 percentage points if images were independent; they are not, so the real uncertainty is larger, and some varieties have as few as 5 test images. The dashboard still uses `models/best.pt`, the model from the original split.

### Running the Evaluation

Activate the project virtual environment and execute:

```bash
python evaluate_rice_classifier.py
```

The evaluation generates confusion matrices and prediction visualizations in the Ultralytics run directory.

`evaluate_rice_classifier.py` also accepts optional `--model`, `--dataset` and `--name` arguments; the defaults reproduce the original evaluation above.

To reproduce the time-ordered evaluation (requires the dataset locally):

`````bash
python make_time_ordered_split.py --buffer 6 --write
python train_rice_classifier.py --dataset rice_classification_time_split_v2 --run-name rice_variety_time_split_v2
python evaluate_rice_classifier.py --model runs/rice_classification/rice_variety_time_split_v2/weights/best.pt --dataset rice_classification_time_split_v2 --name test_v2_time_split
python check_split_leakage.py
````

## 8. Application Features

The CropAI dashboard currently supports:

- Rice variety classification from uploaded images.
- Video analysis using sampled video frames.
- Top-5 classification predictions.
- Prediction confidence scores.
- CSV report generation.
- Maturity planning calculations.

The model classifies the most likely rice variety in the input image. Its confidence score should not be treated as a guarantee of correctness.

## 9. Installation and Setup

Use Python 3.12.10 and create a virtual environment.

```bash
python -m venv .venv
```

Activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## 10. Project Limitations

- Classification performance depends on image quality, lighting, camera angle, and dataset coverage.
- The model predicts a rice variety; it does not identify individual plants.
- The application does not currently provide validated crop-health diagnosis.
- Confidence scores are model estimates, not guaranteed probabilities of correctness.
- The original test split overlaps with the training split (near-identical photos from the same sittings), so its 99.07% result overstates performance on new photos. The stricter time-ordered result is 83.33% Top-1.
- Each variety was photographed in one short sitting, according to file-name timestamps, so neither evaluation measures performance on other days, fields, cameras or lighting. A truly independent test set would need new photographs.
- The time-ordered test set has 288 images (as few as 5 per variety), so its results are statistically noisy.

## 11. Future Improvements

Potential improvements include:

- Testing with more real-world field images and videos.
- Improving robustness under different lighting and environmental conditions.
- Analyzing class-wise errors and improving difficult classifications.
- Exploring crop-health analysis as a separate, properly validated capability.
- Expanding the dataset with additional rice varieties.

## 12. Repository Structure

```text
CropAI/
├── app.py
├── requirements.txt
├── train_rice_classifier.py
├── evaluate_rice_classifier.py
├── models/
│   └── best.pt
├── docs/
│   ├── MODEL_DOCUMENTATION.md
│   └── images/
│       └── confusion_matrix_test.png
└── datasets/
    └── rice_classification_clean_v1/
```

The dataset and local training artifacts are excluded from the GitHub repository.

---

**Documentation status:** Training, validation, and two test evaluations (an original random split and a stricter time-ordered split) have been completed and documented. Test results are based on the current dataset and should not be interpreted as guaranteed real-world performance.