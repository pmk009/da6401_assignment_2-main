# DA6401 Assignment 2: Visual Perception Pipeline

**W&B Report:** (https://wandb.ai/DA6401_JM2026/da6401_assignment2/reports/DA6401-Assignment-02--VmlldzoxNjQzMjY3OA?accessToken=csz2ayjgc8xyx4bqaerisfhws1x7zm8il098iwzkr385oxeomrxcs8hyyk9ihq36) 
**GitHub Repository:** [ADD_LINK_HERE]

---

## Overview

This project implements a multi-task visual perception pipeline using PyTorch on the Oxford-IIIT Pet dataset :contentReference[oaicite:0]{index=0}. The system jointly performs classification, localization, and segmentation within a unified architecture.

---

## Tasks

- **Classification:** 37-class pet breed classification using VGG11  
- **Localization:** Bounding box regression predicting (xc, yc, w, h)  
- **Segmentation:** U-Net style model for trimap prediction  

---

## Architecture

- **Backbone:** VGG11 implemented from scratch  
- **Multi-task Design:** Shared encoder with task-specific heads:
  - Classification head (fully connected layers)
  - Localization head (regression)
  - Segmentation head (encoder-decoder with skip connections)

---

## Key Components

- Custom Dropout layer  
- Custom IoU loss function  
- Batch Normalization for training stability  
- Multi-task learning with a shared feature extractor  

---

## Evaluation Metrics

| Task           | Metric                 |
|----------------|------------------------|
| Classification | Macro F1 Score         |
| Localization   | IoU                    |
| Segmentation   | Dice Score             |

---

## Observations

- Classification shows strong generalization performance  
- Localization exhibits a bias toward discriminative regions (e.g., head) rather than full object extent  
- Segmentation captures global structure but degrades under complex backgrounds and varying lighting conditions  

---

## Generalization to Unseen Data

Evaluation on external images indicates:
- Classification remains reliable  
- Bounding boxes are often misaligned or incomplete  
- Segmentation quality decreases under domain shift  

---

## Usage

```bash
pip install -r requirements.txt
python train.py