# Brain Tumor Segmentation Using U-Net on BraTS2020 Dataset

## 1. Introduction and Research Context

Brain tumor segmentation is a critical step in neuro-oncology for diagnosis, treatment planning, and longitudinal assessment. Manual delineation of tumor subregions in 3D MRI is time-consuming and subject to inter-observer variability. Automated segmentation using deep learning addresses this challenge by providing fast, reproducible, and high-accuracy results.

This study explores 2D deep convolutional segmentation using a U-Net architecture on the BraTS2020 dataset, focusing on multi-class segmentation of tumor subregions.

### Research Objectives

* To develop an end-to-end segmentation pipeline using only the FLAIR and T1ce modalities to reduce memory and computational overhead.
* To train and evaluate a 2D U-Net model for segmenting enhancing, edema, and necrotic core tumor regions.
* To analyze performance using per-class Dice scores and advanced visualization.
* To optimize a deep learning pipeline suitable for real-time or embedded deployment in clinical setups.

### Key Research Questions

* How well can a 2D U-Net segment tumor subregions using only two modalities?
* Can we achieve strong generalization using lightweight data generators and reduced input dimensions?
* Which tumor regions remain the most challenging to segment, and why?

---

## 2. Dataset Description

* **Dataset**: BraTS2020 from the MICCAI Brain Tumor Segmentation Challenge
* **Modality Channels**: T1, T1ce, T2, FLAIR (3D NIfTI volumes)
* **Classes**:

  * 0: Background
  * 1: Necrotic and non-enhancing tumor core
  * 2: Edema
  * 4: Enhancing tumor

Class 4 was relabeled as 3 for computational ease. All images were rescaled using MinMax normalization to \[0,1]. Slice ranges between 22 and 122 were retained based on visual information content.

![__results___26_1](https://github.com/user-attachments/assets/ca750876-9167-4571-bddc-ea43c6a0f3d3)

![__results___36_0](https://github.com/user-attachments/assets/24f6ca2e-5de4-4ef4-9ce6-c99cecf885f3)

![__results___42_0](https://github.com/user-attachments/assets/a9f5eeaa-85f8-4476-bc3a-0d80beb7cd0d)

![__results___47_0](https://github.com/user-attachments/assets/58e22fc6-dfd6-4085-a301-d5341772e73b)

---

## 3. Methodology

### 3.1 Preprocessing Pipeline

* Extract T1ce and FLAIR from 3D NIfTI volumes
* Normalize each modality using MinMaxScaler
* Resize each 2D slice to (128×128)
* Select axial slices 22 to 122 per sample
* One-hot encode the mask with 4 classes (0-3)
* Data split: 68% training, 20% validation, 12% test

### 3.2 Data Generator

Efficient on-the-fly loading and preprocessing using a custom Keras `Sequence`-based generator. Avoids memory overload and preserves patient-level consistency.

---

## 4. Model Architecture

### U-Net (2D)
![download](https://github.com/user-attachments/assets/689ed35d-aac7-4ba3-a7f5-20d75a031b04)

The U-Net consists of:

* **Encoder**: 4 downsampling blocks with `Conv2D → ReLU → Conv2D → MaxPool`
* **Bottleneck**: Dropout layer with 512 filters
* **Decoder**: 4 upsampling blocks with `UpSampling2D → Conv2D → Concatenate → Conv2D`
* **Output**: `Conv2D` with `softmax` activation for 4-class prediction

Input shape: `(128, 128, 2)`
Output shape: `(128, 128, 4)`

### Parameters

* Kernel initializer: `he_normal`
* Dropout rate: `0.2`
* Optimizer: `Adam` with learning rate `1e-3`
* Loss: `Categorical Crossentropy`

---

## 5. Evaluation Metrics

* **Overall**: Accuracy, Mean IoU, Dice Coefficient
* **Per Class Dice**:

  * Dice (Core)
  * Dice (Edema)
  * Dice (Enhancing)
* **Additional**: Precision, Sensitivity, Specificity

---

## 6. Training and Validation

* Epochs: 35
* Batch Size: 1 (volume-wise slicing)
* Callbacks: `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger`

### Training Curves

Training/Validation:

* Accuracy: Converged > 99.3%
* Dice Coefficient: Reached 0.64 macro average
* Mean IoU: Stabilized around 0.84

![Training Curves](attached_curve_example.png)

---

## 7. Results

### Test Performance
![__results___71_0](https://github.com/user-attachments/assets/d5bfc021-580a-4f17-adba-ff43da1bebd9)

| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 0.9931 |
| Mean IoU             | 0.8426 |
| Dice Coefficient     | 0.6480 |
| Dice - Necrotic Core | 0.5916 |
| Dice - Edema         | 0.7667 |
| Dice - Enhancing     | 0.7395 |
| Precision            | 0.9935 |
| Sensitivity          | 0.9916 |
| Specificity          | 0.9978 |

### Visualizations

* Overlay of ground truth and predicted segmentation
* Per-class heatmaps
* Failure analysis on edge slices
* 
![__results___88_2](https://github.com/user-attachments/assets/63b4b6cb-3f78-41e1-b38b-9d5c9977b7dc)

![__results___91_1](https://github.com/user-attachments/assets/da3ebb26-57b0-4a62-9e14-30e5513827a0)

![__results___94_2](https://github.com/user-attachments/assets/cb1e605d-8afc-4dc7-8ead-6fbd200adeca)

---

## 8. Conclusion

This work demonstrates a robust pipeline for brain tumor segmentation using 2D U-Net trained on BraTS2020. It provides:

* High Dice and IoU values with low false positives
* Efficient learning with limited input channels (T1ce + FLAIR)
* Clear interpretability through modular architecture and visualizations

### Limitations and Future Work

* Integration of 3D U-Net for spatial continuity
* Transfer learning from pre-trained medical backbones
* Post-processing using CRFs or edge refinement techniques
* Real-time implementation on portable devices

---

