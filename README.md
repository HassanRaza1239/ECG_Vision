# ECG-Vision: Explainable, Uncertainty-Aware ECG Image Classification

ECG-Vision is an end-to-end deep-learning framework for **four-class ECG image classification**. The project combines image preprocessing, data balancing, transfer learning, feature-level fusion, decision-level ensemble learning, uncertainty estimation, reliability-aware abstention, explainability, robustness testing, shortcut-learning analysis, calibration, OOD analysis, error analysis, and an inference dashboard.

The project is implemented primarily with **TensorFlow/Keras**, OpenCV, NumPy, Pandas, Matplotlib, and scikit-learn, and was developed in a local Jupyter environment using a university **NVIDIA RTX 3070 GPU**.

> **Important:** This repository describes the implementation and results recorded in the project notebook. It is a research/academic system, not a clinically validated medical diagnostic device.

---

## 1. Project Overview

The system classifies an ECG image into one of four categories:

1. **Myocardial Infarction (MI)**
2. **History of MI**
3. **Abnormal heartbeat**
4. **Normal**

The main pipeline is:

```text
Raw ECG image
      │
      ▼
HSV-based red ECG-grid detection
      │
      ▼
ECG-region cropping
      │
      ▼
Train / validation / test organization
      │
      ▼
Training-only class balancing with augmentation
      │
      ▼
Resize to 300 × 300 × 3
      │
      ▼
Pixel scaling to 0–1
      │
      ▼
Train multiple CNN / transfer-learning architectures
      │
      ├───────────────┐
      ▼               ▼
Feature fusion     Standalone models
      │               │
      └───────┬───────┘
              ▼
      Four-model soft-voting ensemble
              │
        ┌─────┼─────────┐
        ▼     ▼         ▼
   Confidence Consensus Uncertainty
              │
              ▼
       Reliability engine
          /         \
      ACCEPT       NEEDS REVIEW
              │
              ▼
 Explainability + robustness + reliability analyses
```

---

## 2. Dataset

### Dataset source

The notebook uses the ECG dataset through a **local `ECG_DATA` directory**. The notebook explicitly does not use Google Drive as the active processed-dataset source.

Typical project path used in the notebook:

```text
/home/hassan_raza/Downloads/ECG/ECG_DATA
```

The notebook does **not record an external dataset URL**, so this README does not invent one. If an official dataset URL is required for publication or submission, it should be added after verifying the exact source used by the project.

### Four classes

The source directory names recorded by the notebook are:

```text
ECG Images of Myocardial Infarction Patients (240x12=2880)
ECG Images of Patient that have History of MI (172x12=2064)
ECG Images of Patient that have abnormal heartbeat (233x12=2796)
Normal Person ECG Images (284x12=3408)
```

### Dataset organization used for modeling

The project creates processed directories such as:

```text
ECG_DATA_PROCESSED/
├── train/
├── val/
└── test/
```

The final generator check reported:

- **Training:** 3,056 images
- **Validation:** 607 images
- **Test:** 928 images

The training set is balanced to **764 images per class**. Validation and test data are not artificially balanced in the same way.

### Train / validation / test meaning

- **Training set:** used to learn model parameters.
- **Validation set:** clean, unaugmented data used while developing/training the models and for selecting reliability thresholds.
- **Test set:** held-out data used for final evaluation.

The project follows the important ordering:

```text
Crop → split → balance training set only → train
```

This avoids using augmented copies of training examples to create the validation/test sets.

---

## 3. ECG Grid Cropping

The preprocessing pipeline detects the ECG print/grid using **HSV color-space thresholding and contour detection**.

### Method

```text
BGR image
   ↓
Convert BGR → HSV
   ↓
Threshold red pixels
   ↓
Combine red hue ranges
   ↓
Find contours
   ↓
Select largest contour
   ↓
Get bounding rectangle
   ↓
Crop slightly inside the border
```

OpenCV is used for the operation.

Two hue ranges are used because red wraps around the HSV hue scale:

```text
Range 1: H 0–10
Range 2: H 160–180
```

with saturation/value constraints used to suppress weak/non-red pixels.

The largest detected red contour is treated as the ECG grid/border. A small padding is removed from the bounding rectangle so that the red border itself is not retained unnecessarily.

This preprocessing is intended to remove irrelevant regions such as patient ID/header/footer material and focus the classifier on the ECG region.

---

## 4. Image Preprocessing and Balancing

### Final model input size

All images are resized to:

```text
300 × 300 × 3
```

The notebook records the **final model input size**, but it does **not record a single numerical original raw-image resolution**. Therefore, the project documentation should not claim that the raw images originally had a specific resolution unless the raw dataset is inspected separately.

The transformation is:

```text
Original image
   ↓
ECG-grid crop
   ↓
Resize → 300 × 300
   ↓
RGB 3-channel tensor
   ↓
Scale pixel values 0–255 → 0–1
```

### Normalization

The generators use:

```python
rescale=1.0 / 255
```

so pixel values are scaled to approximately **0–1**.

Training data also uses a contrast-enhancement preprocessing function with factor `1.5`.

### Training augmentation

The training generator applies small image transformations, including:

- rotation up to ±5°
- width/height shifts
- shear
- zoom
- brightness variation
- contrast enhancement

### Class balancing

Minority classes are physically oversampled **only in the training split**.

Synthetic training examples are generated using small transformations such as:

```text
rotation: approximately −5° to +5°
horizontal flip: probabilistic
```

The target is the size of the largest training class, producing:

```text
764 images × 4 classes = 3,056 training images
```

The project does not balance the test set in this manner.

---

## 5. Implementation Details

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Python | 3.13.5 |
| TensorFlow | 2.21.0 |
| Input size | 300 × 300 × 3 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Random seed | 42 |
| Primary hardware | NVIDIA RTX 3070 GPU |
| Active dataset | Local `ECG_DATA` |
| Model/result backup | Google Drive via PyDrive2 |

The notebook also creates local folders for models, results, plots, Grad-CAM outputs, error analysis, and reports.

Typical structure:

```text
ECG/
├── ECG_DATA/
├── ECG_DATA_PROCESSED/
├── models/
├── results/
├── plots/
├── gradcam/
├── error_analysis/
└── reports/
```

---

## 6. Model Architectures

The notebook evaluates the following standalone architectures:

1. **Custom CNN** — trained from scratch (baseline)
2. **VGG16** — ImageNet-pretrained transfer learning
3. **DenseNet121** — ImageNet-pretrained transfer learning
4. **ResNet50** — ImageNet-pretrained transfer learning
5. **EfficientNetB0** — ImageNet-pretrained transfer learning
6. **InceptionV3** — ImageNet-pretrained transfer learning
7. **MobileNetV2** — ImageNet-pretrained transfer learning
8. **ConvNeXt** — transfer-learning architecture
9. **Vision Transformer (ViT)**

### Pretrained vs from-scratch

A **pretrained** architecture starts from weights learned on a previous dataset (ImageNet in the transfer-learning models) and is adapted to the ECG task.

The **Custom CNN** starts with randomly initialized weights and learns its representation from the ECG training data.

---

## 7. Two-Stage Transfer Learning

The transfer-learning models follow a two-stage strategy.

### Stage 1 — Frozen backbone

The pretrained feature extractor/backbone is frozen:

```text
Backbone → frozen
New classifier head → trainable
```

The classifier learns how to map the extracted features to the four ECG classes.

### Stage 2 — Fine-tuning

After initial training, the backbone is opened for controlled fine-tuning at a smaller learning rate.

Batch Normalization layers are kept frozen in this stage to improve training stability.

Conceptually:

```text
Stage 1:
Pretrained backbone 🔒 + trainable classification head

Stage 2:
Partially/fully trainable backbone + classification head
with lower learning rate
```

This lets the model retain useful general visual features while adapting them to ECG-specific patterns.

---

## 8. Features, Parameters, and Penultimate Layers

### Features

A neural network transforms the image through many layers into numerical representations. These learned representations are the model's **features**.

Conceptually:

```text
ECG image
   ↓
low-level edges / shapes
   ↓
waveform-related structures
   ↓
high-level representation
   ↓
classification
```

### Parameters

Model parameters are learned numerical values such as:

- weights
- biases
- convolutional filter values

### Hyperparameters

Hyperparameters are chosen by the experimenter, such as:

- learning rate
- batch size
- number of epochs
- dropout rate
- image size
- augmentation strength

### Penultimate layer

The **penultimate layer** is the second-to-last layer before the final classification layer.

The final layer outputs four class probabilities, while the penultimate layer contains a learned feature representation. The project uses these feature representations in the fusion models.

---

## 9. Feature-Level Fusion

Two fusion architectures are used:

```text
DenseNet121 + VGG16
DenseNet121 + ResNet50
```

Their feature representations are extracted from the penultimate layers and concatenated into a joint feature vector.

Conceptually:

```text
                 ECG image
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     DenseNet121              VGG16/ResNet50
          ▼                     ▼
     Feature vector A       Feature vector B
          └──────────┬──────────┘
                     ▼
                Concatenate
                     ▼
                Fusion head
                     ▼
              4-class Softmax
```

The pretrained source backbones are used as feature extractors while the newly added fusion classification layers are trained for the combined representation.

---

## 10. Final Ensemble: Decision-Level Soft Voting

The final ensemble contains **four selected models**:

```text
1. DenseNet121
2. ResNet50
3. Fusion: DenseNet+VGG
4. Fusion: DenseNet+ResNet
```

The other standalone models are still evaluated individually, but their predictions are **not** members of the final four-model soft-voting ensemble.

### Soft voting

Each ensemble member produces a four-class probability vector.

The ensemble averages those vectors:

```text
ensemble_probs = mean(model_probability_vectors)
```

The class with the highest averaged probability is selected.

This is **soft voting**, not simple majority counting.

---

## 11. Confidence, Consensus, and Uncertainty

The framework computes three independent signals for a prediction.

### 11.1 Confidence

Confidence is the ensemble probability assigned to its predicted class:

```text
confidence = max(ensemble_probabilities)
```

Example:

```text
MI            0.95
History MI    0.02
Abnormal      0.02
Normal        0.01
```

Confidence = `0.95` or **95%**.

### 11.2 Consensus

Consensus measures how many of the four ensemble members agree on the same predicted class.

```text
4 / 4 models agree → 1.00 = 100%
3 / 4 models agree → 0.75 = 75%
2 / 4 models agree → 0.50 = 50%
```

### 11.3 Monte Carlo Dropout uncertainty

The project estimates uncertainty using **Monte Carlo Dropout (MC Dropout)**.

Dropout is kept active during inference and the same image is evaluated multiple times. The spread of the resulting probability vectors is measured using standard deviation.

Conceptually:

```text
Same ECG image
   ↓
20 stochastic predictions (single-image inference)
   ↓
probability variation
   ↓
mean per-class standard deviation
   ↓
uncertainty score
```

Lower spread means more stable predictions; higher spread indicates more prediction variation.

---

## 12. Reliability Engine and Abstention

The project does not rely on confidence alone.

A prediction is accepted only when all three conditions pass:

```text
confidence >= confidence threshold
AND
consensus >= consensus threshold
AND
uncertainty < uncertainty threshold
```

Otherwise the framework returns:

```text
NEEDS REVIEW
```

### Thresholds used in the recorded run

| Signal | Threshold |
|---|---:|
| Confidence | 1.0000 |
| Consensus | 0.75 |
| Uncertainty | 0.0000 |

The notebook derives the confidence and uncertainty thresholds from **validation data**, not from the final test set. The consensus threshold is fixed at 0.75.

### Recorded reliability result

```text
Test images:       928
Accepted:          522 (56.2%)
Needs review:      406 (43.8%)
Accepted accuracy: 100.00%
Review accuracy:   99.26%
Overall accuracy:  99.68%
```

The 100% accepted accuracy should be interpreted as an **abstention/reliability result on a filtered subset**, not as evidence that the complete test set is perfect.

---

## 13. Explainability

### Grad-CAM

Grad-CAM is used to visualize which image regions contribute strongly to a model's prediction.

The project generates multi-model Grad-CAM outputs and compares the activated regions using pairwise IoU overlap.

### Occlusion sensitivity

Occlusion sensitivity moves a patch over the ECG image, hides that region, and measures how the prediction changes.

The project uses approximately:

```text
Patch size: 30
Stride:     15
```

A large confidence change after hiding a region suggests that the region is influential for the prediction.

The notebook's sampled Grad-CAM/occlusion comparisons show that the two explanation methods do not always strongly agree, which is documented as an interpretability limitation rather than being treated as proof of a single "correct" explanation map.

---

## 14. Robustness Testing

Robustness testing evaluates the primary DenseNet121 model after controlled image degradation.

The notebook evaluates:

- Original
- Brightness
- Darkness
- Contrast
- Gaussian noise
- Blur
- Rotation
- JPEG compression
- Occlusion

Recorded results:

| Condition | Accuracy | Macro F1 |
|---|---:|---:|
| Original | 99.68% | 99.71% |
| Brightness | 99.46% | 99.47% |
| Darkness | 98.60% | 98.63% |
| Contrast | 99.78% | 99.77% |
| Gaussian Noise | 99.68% | 99.71% |
| Blur | **61.85%** | **59.47%** |
| Rotation | 93.10% | 93.17% |
| Compression | 97.31% | 97.56% |
| Occlusion | 98.60% | 98.55% |

The major observed weakness is **blur**, followed by sensitivity to larger rotations.

---

## 15. Shortcut-Learning Analysis

Shortcut learning occurs when a model uses an easier but irrelevant visual cue instead of the intended signal.

To investigate this, the project compares:

```text
1. Original image
2. ECG-only image (waveform isolated)
3. Background-only image (waveform removed)
```

Recorded results:

| Input version | Accuracy | Macro F1 |
|---|---:|---:|
| Original | 99.68% | 99.71% |
| ECG-only | 99.57% | 99.57% |
| Background-only | 25.75% | 10.24% |

The background-only performance is close to the four-class chance level, while ECG-only performance remains very high. This supports the interpretation that the model is strongly using ECG information rather than relying mainly on background artifacts.

---

## 16. Calibration Analysis

Calibration asks whether predicted confidence reflects empirical correctness.

A well-calibrated classifier should behave approximately like this:

```text
90% confidence → about 90% correct
```

The project calculates **Expected Calibration Error (ECE)**.

Recorded result:

```text
ECE = 0.0039
```

Lower ECE is better; zero corresponds to perfect calibration under the metric definition used in the notebook.

---

## 17. Out-of-Distribution (OOD) Detection

OOD detection asks whether the system can recognize inputs that do not resemble the training distribution.

The project evaluates confidence behavior using **Maximum Softmax Probability (MSP)** and compares in-distribution ECG inputs against OOD examples.

The notebook's final wrap-up confirms that OOD analysis is part of the framework, but the expected saved file:

```text
results/predictions/ood_results.csv
```

was **not present at the final checkpoint**. Therefore this README intentionally does not report a numerical OOD score that could not be verified from the recorded outputs.

---

## 18. Error Analysis

For DenseNet121, the recorded clean test evaluation was:

```text
Total test images: 928
Correct:           925
Misclassified:     3
```

The average confidence was:

```text
Correct predictions:   99.72%
Incorrect predictions: 83.12%
```

The recorded misclassification pattern shows abnormal-heartbeat cases confused with normal ECGs.

This analysis is useful because it identifies where the model still fails instead of focusing only on aggregate accuracy.

---

## 19. Final Model Comparison

The notebook evaluates 12 final comparison entries: 9 standalone architectures, 2 fusion models, and 1 ensemble.

| Model | Type | Accuracy | Precision | Recall | Macro F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| **Fusion: DenseNet+ResNet** | Feature Fusion | **99.78%** | **99.83%** | **99.79%** | **99.80%** | **100.00%** |
| InceptionV3 | Transfer Learning | 99.78% | 99.79% | 99.79% | 99.79% | 100.00% |
| VGG16 | Transfer Learning | 99.78% | 99.79% | 99.71% | 99.75% | 100.00% |
| DenseNet121 | Transfer Learning | 99.68% | 99.74% | 99.68% | 99.71% | 100.00% |
| Fusion: DenseNet+VGG | Feature Fusion | 99.68% | 99.74% | 99.68% | 99.71% | 100.00% |
| Ensemble (Soft-Voting) | Ensemble | 99.68% | 99.74% | 99.68% | 99.71% | 99.99% |
| ResNet50 | Transfer Learning | 99.68% | 99.68% | 99.68% | 99.68% | 99.98% |
| Custom CNN | Baseline | 99.68% | 99.68% | 99.68% | 99.68% | 100.00% |
| ConvNeXt | Transfer Learning | 99.68% | 99.62% | 99.68% | 99.65% | 100.00% |
| MobileNetV2 | Transfer Learning | 95.26% | 94.80% | 95.51% | 94.88% | 99.80% |
| EfficientNetB0 | Transfer Learning | 50.00% | 40.82% | 44.83% | 36.08% | 72.46% |
| Vision Transformer | Transfer Learning | 25.75% | 6.44% | 25.00% | 10.24% | 60.21% |

### Best recorded model

The best overall entry by macro F1 is:

```text
Fusion: DenseNet+ResNet
Accuracy:  99.78%
Precision: 99.83%
Recall:    99.79%
Macro F1:  99.80%
AUC:       100.00%
```

---

## 20. Ensemble vs Individual Models

The final ensemble uses only:

```text
DenseNet121
ResNet50
DenseNet+VGG fusion
DenseNet+ResNet fusion
```

The remaining standalone models are still important because they provide the architecture comparison used to select strong candidates and demonstrate how different architectures behave on the same ECG classification task.

The ensemble should therefore be understood as a **selected-model ensemble**, not an ensemble of every architecture that was trained.

---

## 21. Ablation Study

The notebook progressively measures the contribution of major framework components:

| Experiment | Accuracy | Macro F1 | Interpretation |
|---|---:|---:|---|
| A: DenseNet121 baseline | 99.68% | 99.71% | Clean test-set baseline |
| B: + DenseNet+ResNet fusion | 99.78% | 99.80% | Best feature-fusion model |
| C: + soft-voting ensemble | 99.68% | 99.71% | Selected four-model ensemble |
| D: + uncertainty filtering | 100.00% | 100.00% | Low-uncertainty subset only |
| E: + abstention | 100.00% | 100.00% | ACCEPT-only subset |
| F: robustness | 93.55% | 93.29% | Average over degraded conditions |

**Important interpretation:** Experiments D and E use progressively smaller subsets. Their 100% accuracy does not mean that the entire test set became perfect; it means that the retained/reliable subset was completely correct in the recorded run. Experiment F uses degraded inputs and answers a different robustness question.

---

## 22. Founder / Original Baseline Comparison

The notebook also compares the project implementation with a supplied/founder result.

| Evaluation | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Founder / Original | 85.78% | 87.18% | 85.02% | Not reported | 97.46% |
| Our DenseNet121 | 99.68% | 99.74% | 99.68% | 99.71% | 100.00% |

The notebook explicitly warns that this is **not a strict apples-to-apples algorithm comparison**, because the methodologies differ. Documented differences include ECG-grid cropping, physical class balancing instead of class weighting, a clean validation pipeline, random seed/environment differences, and hardware differences.

---

## 23. Why the Results Are Very High

The notebook produces unusually high performance on the clean test set. This should be interpreted carefully.

Possible contributors documented by the project methodology include:

- The dataset can be visually separable.
- ECG-grid cropping removes irrelevant regions.
- Training data is balanced.
- Transfer-learning models start with pretrained visual representations.
- The test distribution comes from the same dataset family and image style as the development pipeline.
- The reliability/abstention analysis evaluates a filtered subset when reporting 100% accepted accuracy.

Therefore, the reported **99.68–99.78% clean-test performance should not be interpreted as clinical-level generalization** to different hospitals, scanners, patient populations, acquisition styles, or unseen datasets.

External validation is still required before any clinical claim could be made.

---

## 24. Single-Image Inference Dashboard

The notebook contains a full single-image inference pipeline:

```text
Input image
   ↓
ECG-grid crop detection
   ↓
Resize + normalization
   ↓
Four ensemble-member predictions
   ↓
Soft-voting prediction
   ↓
Consensus
   ↓
MC-Dropout uncertainty
   ↓
Reliability decision
   ↓
Grad-CAM
   ↓
Image-quality assessment
   ↓
Dashboard output
```

A key safety feature is the **non-ECG guard**: when the red ECG grid border is not detected, the dashboard can reject the input as likely non-ECG instead of treating the classifier's forced four-class softmax prediction as trustworthy.

The notebook demonstrated this behavior on a non-ECG-like input, where the image was rejected despite the classifier still producing a class probability vector.

---

## 25. Model / Result Files

The notebook records a number of local artifact locations, including:

```text
models/
├── custom_cnn.keras
├── fusion_densenet_vgg.keras
├── fusion_densenet_resnet.keras
└── ...

results/
├── metrics/
│   ├── final_model_comparison.csv
│   ├── final_model_comparison_formatted.csv
│   ├── founder_comparison.csv
│   ├── reliability_thresholds.json
│   └── calibration-related metrics
│
├── predictions/
│   ├── ensemble_cache.npz
│   ├── consensus_scores.csv
│   ├── uncertainty_scores.csv
│   ├── reliability_decisions.csv
│   ├── gradcam_overlap.csv
│   └── shortcut_learning_analysis.csv
│
├── robustness/
│   └── robustness_results.csv
│
└── checkpoints/
    └── ...
```

The exact set of files present can vary depending on which notebook cells have been executed and cached.

---

## 26. Reproducibility

The notebook sets:

```python
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

Model and result caching is used extensively. Several cells ask whether previously saved results should be reused rather than recomputing full test-set evaluations.

Training and inference may still show small environmental differences because GPU execution, library versions, and other system-level factors can introduce numerical variation.

---

## 27. Installation / Environment

The project notebook records packages for the local Jupyter environment, including:

```text
TensorFlow / Keras
OpenCV (cv2)
NumPy
Pandas
Matplotlib
scikit-learn
PyDrive2
vit-keras (where supported by the environment)
```

A minimal conceptual installation is:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python tqdm PyDrive2
```

The notebook environment also contains support for the tested transformer/transfer-learning components. Exact package compatibility should follow the versions used by the notebook rather than assuming a generic environment will reproduce every model identically.

---

## 28. Running the Notebook

A typical execution order is:

1. Start Jupyter in the project root.
2. Run setup and GPU verification.
3. Verify the local `ECG_DATA` directory.
4. Run ECG-grid cropping and dataset preprocessing.
5. Build the processed train/validation/test directories.
6. Create data generators.
7. Train or load the standalone models.
8. Build/train/load the two feature-fusion models.
9. Evaluate the selected-model soft-voting ensemble.
10. Run consensus and MC-Dropout uncertainty analysis.
11. Run reliability/abstention analysis.
12. Generate Grad-CAM and occlusion analyses.
13. Run robustness, shortcut-learning, calibration, OOD, and error analyses.
14. Run the ablation and comparison cells.
15. Use the single-image dashboard for inference.

Because the notebook supports cached artifacts, later runs can reuse previously saved models and results where available.

---

## 29. Limitations

The recorded results should be interpreted with the following limitations:

- The dataset is from a single source/distribution and is not an external multi-site clinical benchmark.
- The notebook does not document a verified external dataset URL.
- The original raw image dimensions are not recorded as one fixed value in the notebook.
- Robustness drops substantially under blur.
- The sampled Grad-CAM and occlusion explanations do not always overlap strongly.
- The uncertainty threshold in the recorded run is `0.0000`, reflecting a highly concentrated uncertainty distribution and deserving further calibration before practical deployment.
- OOD results are part of the intended framework, but the final saved `ood_results.csv` was missing at the recorded checkpoint.
- The founder comparison is methodologically different and should not be treated as a controlled head-to-head benchmark.
- Very high clean-test scores should not be generalized to real-world clinical performance without external validation.

---

## 30. Project Status

The notebook records the project pipeline as complete, with the major components implemented and evaluated:

```text
[✓] Local dataset pipeline
[✓] ECG-grid cropping
[✓] Training-only balancing
[✓] Multi-model training/evaluation
[✓] Feature fusion
[✓] Four-model soft voting
[✓] Consensus analysis
[✓] MC-Dropout uncertainty
[✓] Reliability / abstention
[✓] Grad-CAM
[✓] Occlusion sensitivity
[✓] Robustness testing
[✓] Shortcut-learning analysis
[✓] Calibration analysis
[✓] Error analysis
[✓] Ablation study
[✓] Single-image dashboard
[~] OOD artifact requires verification/rerun
```

---

## 31. Research Questions Addressed

The notebook frames the project around questions including:

- Does DenseNet121 + ResNet50 feature fusion improve over individual models?
- Does selected-model ensemble learning improve performance/stability?
- Can model disagreement identify unreliable predictions?
- Can uncertainty estimation identify untrustworthy predictions?
- Does abstention reduce unreliable automated predictions?
- Do different models focus on similar ECG regions?
- How robust is the model to image degradation?
- Does the model rely on ECG information rather than background artifacts?
- Can OOD analysis identify unusual inputs?
- Does the complete framework add value beyond a single classifier?

---

## 32. Team / Academic Project

**Project:** ECG-Vision — Explainable, Uncertainty-Aware Ensemble Framework

**Environment:** Local Jupyter / University GPU server

**Primary framework:** TensorFlow / Keras

**Primary model:** DenseNet121

**Best recorded model:** DenseNet121 + ResNet50 feature fusion

**Final selected ensemble:** DenseNet121 + ResNet50 + DenseNet/VGG fusion + DenseNet/ResNet fusion

---

## 33. Citation / Dataset Attribution

The notebook does not has citations.

---

## License

No project license is specified.
