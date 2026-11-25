# Aerial-Human-Detection
Scale and Rotation-Robust Aerial Human Detection

# Scale and Rotation-Robust Aerial Human Detection

**Alex Eagles Tech | Computer Vision Task 3**

## 1. Project Overview

This project investigates human detection under challenging aerial imaging conditions, including small object sizes, varying camera orientations, illumination changes, and different viewing angles.

The objective is to develop a robust object detection pipeline for a 590-image competition benchmark without an official training dataset.

Rather than relying solely on a pretrained detector, the project follows an experimental approach to improve detection performance through high-resolution inference, confidence threshold optimization, domain-specific fine-tuning, and planned inference enhancement techniques.

## 2. Key Challenges

- Small and distant human detection.
- Arbitrary camera rotations.
- Variations in brightness, contrast, and color balance.
- Domain shift between conventional and aerial imagery.
- Balancing false positives and missed detections.

## 3. Datasets

### COCO

COCO-pretrained YOLO11m weights were used to establish the initial detection baseline.

### VisDrone2019-DET

VisDrone was selected as an external dataset for aerial human detection and domain adaptation.

| Split | Images | Purpose |
|---|---:|---|
| Training | 6,471 | Domain-specific fine-tuning |
| Validation | 548 | Local evaluation and optimization |
| Competition test | 590 | Automated inference and submission |

The pedestrian and people categories in VisDrone were merged into a single human class.

Annotations were converted from absolute pixel coordinates to normalized YOLO bounding box coordinates, with bounding box clipping applied during training label preparation.

The competition test images were excluded from model training and manual annotation.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8449cf69-9b48-4281-9327-e1fe93bd25f5" />

*Example of annotated humans in aerial imagery from the VisDrone validation dataset.*



## 4. Methodology

The project follows an iterative experimental pipeline:

1. Establish a COCO-pretrained YOLO11m baseline.
2. Analyze predicted bounding box sizes and confidence distributions.
3. Evaluate the impact of increased input resolution.
4. Construct an external aerial validation set.
5. Implement a local approximation of the competition evaluation metric.
6. Optimize the confidence threshold through systematic experiments.
7. Fine-tune YOLO11m on VisDrone training images.
8. Investigate rotation-based test-time augmentation and tiled inference.

Each experiment is compared against previous configurations to assess its contribution to detection performance.

## 5. Custom Evaluation Metric

A custom local evaluator was implemented based on the competition's Hard + Clamped Soft F1 specification.

Predicted bounding boxes are matched to ground-truth annotations through greedy one-to-one matching using Intersection over Union (IoU).

The implemented matching credits are:

| IoU range | Hard credit | Soft credit |
|---|---:|---:|
| IoU below 0.15 | 0 | 0 |
| 0.15 ≤ IoU < 0.30 | 0 | IoU / 0.30 |
| IoU ≥ 0.30 | 1 | 1 |

The final score is calculated as:

**Final Score = 0.5 × Hard F1 + 0.5 × Clamped Soft F1**

This implementation enables local comparison of inference configurations without relying exclusively on the public leaderboard.

## 6. Exploratory Data Analysis

The initial prediction analysis revealed that a large proportion of detected objects occupied less than 0.5% of the image area.

<img width="696" height="470" alt="image" src="https://github.com/user-attachments/assets/224a31bf-faae-47ce-bf3c-37664dfd7c29" />

*Distribution of normalized bounding box areas in the initial baseline predictions.*

<img width="686" height="470" alt="image" src="https://github.com/user-attachments/assets/7cf6fb93-3b5e-4a82-a655-99f68a853d45" />

*Confidence score distribution of the pretrained baseline.*

Further analysis of the VisDrone validation annotations showed that approximately **99.39% of annotated humans** satisfied this small-object criterion.

<img width="790" height="470" alt="image" src="https://github.com/user-attachments/assets/e8971b18-0cd6-4287-9d89-7f6f13e02cea" />

*Ground-truth human size distribution in the VisDrone validation set.*

This observation motivated experiments involving increased input resolution.

## 7. Ablation Studies

### Experiment 1 — Pretrained Baseline

- Model: YOLO11m
- Input resolution: 640
- Confidence threshold: 0.25
- NMS IoU threshold: 0.45
- Additional training: None

### Experiment 2 — High-Resolution Inference

The input resolution was increased from 640 to 1280 while maintaining the same model weights and confidence threshold.

| Metric | 640px | 1280px |
|---|---:|---:|
| Hard F1 | 0.265675 | 0.498456 |
| Soft F1 | 0.266906 | 0.500045 |
| Final local score | 0.266290 | 0.499251 |
| Predictions | 2,299 | 5,142 |
| Ground-truth instances | 13,969 | 13,969 |

Increasing the input resolution improved the local blended F1 score by approximately 87.5% relative to the initial baseline.

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/7d0534ba-2192-46d0-b126-69b7b9b04e20" />

*Impact of increasing the input resolution from 640 to 1280 pixels on detection counts.*

### Experiment 3 — Confidence Threshold Optimization

A systematic threshold search was conducted on the VisDrone validation set.

The initial search covered confidence thresholds between 0.05 and 0.50. Since the highest score occurred at the lower boundary, an extended search was conducted.

<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/0d809122-06f8-41bd-943b-eedd76eec07e" />

*Initial confidence threshold search using the local blended F1 metric.*

The best observed confidence threshold was **0.04**, producing a local blended F1 score of **0.633276**.

<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/9711a37a-eb5f-4731-971d-f47b526d1494" />

*Extended threshold search identifying 0.04 as the best tested confidence threshold.*

The results suggest that the initial confidence threshold was overly restrictive for this aerial validation dataset.

### Experiment 4 — Aerial Fine-tuning

Status: In progress.

YOLO11m is being fine-tuned on the VisDrone training split using transfer learning and domain-specific augmentation.

Training configurations include:

- Input resolution: 960
- Training images: 6,471
- Random rotation: ±180 degrees
- Horizontal and vertical flipping
- HSV augmentation
- Scale and translation augmentation
- Mosaic augmentation
- Optimizer: AdamW
- Initial learning rate: 0.0005
- Random seed: 42

The trained model will be evaluated using the custom local metric before selecting the final inference configuration.

## 8. Results

| Experiment | Local Score | Kaggle Score |
|---|---:|---:|
| E1 — Initial baseline | 0.266290 | 0.55109 |
| E2 — High resolution | 0.499251 | Not submitted |
| E3 — Confidence optimization | 0.633276 | **0.70867** |
| E4 — Aerial fine-tuning | Pending | Pending |
| E5 — Rotation TTA | Pending | Pending |
| E6 — Tiled inference | Pending | Pending |

### Current Best Performance

- Initial Kaggle score: 0.55109
- Best Kaggle score: 0.70867
- Absolute improvement: 0.15758
- Relative improvement: approximately 28.6%

These improvements were achieved before completing domain-specific fine-tuning.

<img width="3064" height="423" alt="image" src="https://github.com/user-attachments/assets/1dec596c-b2c6-4927-8816-eaa21eec94ff" />

*Competition score improvement from 0.55109 to 0.70867 through inference optimization.*


## 9. Submission Format

The submission file contains two columns:

- image_id
- PredictionString

Based on the evaluator's observed requirements, each detection is exported using five values:

`class_id x_center y_center width height`

The human class is exported as class ID 1.

Confidence scores are retained internally for filtering but are excluded from the final submission string.

Images without detections are represented using a whitespace string to satisfy the submission parser's non-null requirement.

## 10. Technologies

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Google Colab
- Kaggle Notebooks

## 11. Reproducibility

The experiments use publicly available datasets and pretrained model weights.

Training configurations, preprocessing procedures, evaluation code, and inference parameters are documented in the project notebook.

A fixed random seed is used for training experiments.

## 12. Limitations

- VisDrone does not perfectly represent the competition's image distribution.
- The local evaluator is an approximation based on the published metric description rather than the official evaluator implementation.
- Ignored regions in VisDrone are not yet fully handled by the custom evaluation metric.
- Hyperparameters are optimized on the validation dataset; their performance may differ on unseen data.
- Computational resources and training time limit the number of experiments.

## 13. Planned Improvements

- Rotation-based test-time augmentation.
- Overlapping tiled inference.
- Fine-tuned model confidence calibration.
- Model ensembling and Weighted Boxes Fusion.
- Detailed failure analysis on external labeled validation images.

## 14. References

- Ultralytics: https://github.com/ultralytics/ultralytics
- VisDrone Dataset: https://github.com/VisDrone/VisDrone-Dataset
- COCO Dataset: https://cocodataset.org/
- Competition: https://www.kaggle.com/competitions/human-detection-ae

## Project Status

Work in progress. Current results represent intermediate experiments rather than the final optimized solution.
