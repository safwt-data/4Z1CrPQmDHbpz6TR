<h1 align="center">📖 MonReader Page-Flip Detection</h1>

<p align="center">
A computer vision project that classifies individual video frames as page flipping or not flipping and compares models for mobile document scanning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Computer%20Vision-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange" alt="TensorFlow and Keras">
  <img src="https://img.shields.io/badge/MobileNetV2-Test%20F1%200.924-green" alt="MobileNetV2 test F1 of 0.924">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-blueviolet" alt="Grad-CAM">
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Questions](#-business-questions)
- [Project Workflow](#-project-workflow)
- [1. Data Preparation](#1-data-preparation)
- [2. Preprocessing and Augmentation](#2-preprocessing-and-augmentation)
- [3. Computer Vision Models](#3-computer-vision-models)
- [4. Model Optimization](#4-model-optimization)
- [5. Model Explainability](#5-model-explainability)
- [6. Gradio Application](#8-gradio-application)
- [Key Findings](#-key-findings)
- [Business Interpretation](#-business-interpretation)
- [Project Contribution](#-project-contribution)
- [Conclusion](#-conclusion)

---

## 🔎 Project Overview

MonReader uses video frames from document scanning. Given one RGB frame, the model predicts whether a page is being turned. This prediction could help a scanning app decide when to wait and when a frame is ready for further checks.

The project compares a custom CNN with ImageNet-pretrained models and measures both classification performance and saved model size. The project guide sets two targets for mobile use — **test F1 above 0.91** and **a saved model under 40 MB**.

## 🎯 Business Questions

1. **Can the model detect a page turn from a single frame?**
2. **Which architecture delivers the highest test F1 within the mobile file-size budget?**
3. **How do augmentation and partial fine-tuning change the results?**
4. **Which image regions contribute to a prediction?**

## 🔄 Project Workflow

**Load and split images → Prepare model inputs → Train baseline and transfer-learning models → Evaluate saved models → Compare F1 and file size → Visualize predictions with Grad-CAM**

---

## 1. Data Preparation

Images are stored in class folders under separate training and testing directories. Keras creates an 80/20 training and validation split from the training directory using seed 42. The test loader keeps a fixed order with `shuffle=False`.

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 1,914 | Fit model weights |
| Validation | 478 | Track training and monitor validation loss |
| Test | 597 | Compare saved models on held-out images |

Keras assigns labels alphabetically. **`flip` is label 1** and **`notflip` is label 0**. At a threshold of 0.5, higher sigmoid scores indicate `notflip`. The reported positive-class precision, recall, and F1 therefore describe `notflip`.

## 2. Preprocessing and Augmentation

- Load RGB images at **224 × 224** in batches of **32**.
- Apply the input preprocessing expected by each pretrained backbone. The notebook includes preprocessing layers for VGG16, ResNet50, and MobileNetV2. EfficientNetB0 handles rescaling internally.
- Compare plain VGG16 with a training-time random contrast variant. The executed augmentation definition uses contrast with a factor of 0.1.

## 3. Computer Vision Models

The custom CNN provides a baseline trained from scratch. The transfer-learning experiments use ImageNet weights and a compact classification head with global average pooling and a sigmoid output.

### Model comparison on the test set

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | PR-AUC | Size (MiB) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 0.9397 | 0.9436 | 0.9094 | 0.9805 | 0.9917 | 0.9886 | 0.89 | 73,825 |
| ResNet50 | 0.9698 | **0.9714** | 0.9474 | 0.9967 | 0.9969 | 0.9971 | 90.63 | 23,589,761 |
| VGG16 | 0.8894 | 0.8978 | 0.8555 | 0.9446 | 0.9620 | 0.9597 | 56.22 | 14,715,201 |
| VGG16 + augmentation | 0.9062 | 0.9111 | 0.8885 | 0.9349 | 0.9701 | 0.9685 | 56.23 | 14,715,201 |
| EfficientNetB0 | 0.9263 | 0.9312 | 0.8949 | 0.9707 | 0.9806 | 0.9822 | 16.27 | 4,050,852 |
| MobileNetV2 (frozen) | 0.9213 | 0.9238 | 0.9194 | 0.9283 | 0.9835 | 0.9847 | 9.19 | 2,259,265 |
| MobileNetV2 (partially unfrozen) | 0.9548 | 0.9577 | 0.9217 | 0.9967 | 0.9963 | 0.9964 | 18.40 | 2,259,265 |

Fine-tuned MobileNetV2 achieved a higher test F1 score than frozen MobileNetV2 (0.9577 vs. 0.9238), but its training and validation loss curves showed signs of overfitting. The frozen model showed a much smaller gap between training and validation loss, indicating more stable validation performance during training. I selected MobileNetV2 with a frozen backbone for the Gradio app because of this stability. It also had a smaller saved file size (9.19 MiB vs. 18.40 MiB).

## 4. Model Optimization

The frozen MobileNetV2 trains the classification head while keeping the pretrained backbone fixed. The partial fine-tuning experiment starts from ImageNet weights with the final **20 backbone layers** trainable. It uses Adam at **0.00001** and early stopping on validation loss with best-weight restoration.

Partial fine-tuning increased notflip recall from 0.9283 to 0.9967 and precision from 0.9194 to 0.9217. Test F1 rose from 0.9238 to 0.9577, and the saved model grew from 9.19 MiB to 18.40 MiB. The fine-tuned model’s loss curves showed signs of overfitting, so I chose frozen MobileNetV2 for the Gradio app because it had a smaller gap between training and validation loss.

## 5. Model Explainability

The notebook uses **Grad-CAM** on the partially unfrozen MobileNetV2 to visualize which regions of an example frame contribute to its predicted class. It weights the final backbone feature maps by the gradients of the class score, normalizes the resulting heatmap, and overlays it on the original frame. 

The side-by-side view helps inspect whether the prediction draws on the moving page edge, the document surface, or surrounding areas

## 6. Gradio Application

Link to article: https://medium.com/@safwtsafwt19/can-a-single-frame-catch-a-page-turn-building-a-classifier-for-mobile-scanning-7ea00ac353e1

The MonReader prototype lets users test page-flip detection on a single image. The interface provides:

- An image upload field for a frame from a page-flipping video.
- A prediction showing whether the page is flipping or not flipping.
- A Grad-CAM overlay highlighting the image regions that contributed to the prediction.

## 🔑 Key Findings

- **ResNet50 achieved the highest test F1.** Its 90.63 MiB saved model exceeds the mobile file-size budget.
- **Frozen MobileNetV2 led the models that met both targets.** It achieved 0.9236 F1 at 9.19 MiB. EfficientNetB0 also met both targets, with 0.9216 F1 at 16.27 MiB.
- **The VGG16 random-contrast run scored lower than augmented VGG16.** Its F1 dropped from 0.8662 to 0.6497 in this comparison.

## 💼 Business Interpretation

Frozen MobileNetV2 is the recommended candidate for mobile integration based on test F1 and saved file size. A false `notflip` prediction can allow a moving frame into the scan pipeline. A false `flip` prediction can delay capture of a usable frame. The app can use these two error costs to decide how conservative its capture threshold should be. The comparison measures saved file size and test accuracy metrics.

## 👨‍💻 Project Contribution

- Built a TensorFlow/Keras pipeline for directory-based image loading, validation splitting, and held-out testing.
- Trained a custom CNN and compared four pretrained architectures across seven model configurations.
- Evaluated transfer learning, random contrast augmentation, and partial backbone fine-tuning.
- Saved and reloaded Keras models and compared test metrics, parameter counts, and file sizes.
- Implemented a Grad-CAM overlay to inspect a prediction from the tuned MobileNetV2.


## ✅ Conclusion

Frozen MobileNetV2 is the strongest model within the project’s mobile size limit. Its **0.9236 test F1 at 9.19 MiB** meets both project targets and provides a candidate for testing in the document-scanning pipeline.
