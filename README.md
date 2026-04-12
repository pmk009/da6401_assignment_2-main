# Assignment 2: Visual Perception Pipeline

**Name:** Kishore M  
**Roll Number:** AE21B036  
**Course:** DA6401 - Introduction to Deep Learning (Jan-May 2026)

---

## Weights & Biases Report

All experiments, ablations, and evaluations are logged using **Weights & Biases**.

W&B Report:  
https://wandb.ai/DA6401_JM2026/da6401_assignment2/reports/DA6401-Assignment-02--VmlldzoxNjQzMjY3OA?accessToken=csz2ayjgc8xyx4bqaerisfhws1x7zm8il098iwzkr385oxeomrxcs8hyyk9ihq36

---

## GitHub Repository

Project Repository:  
https://github.com/pmk009/da6401_assignment_2-main

---

## Overview

This project implements a **multi-task visual perception pipeline using PyTorch**.  
The system integrates classification, localization, and segmentation into a unified architecture based on a shared VGG11 backbone.

The model is trained and evaluated on the **Oxford-IIIT Pet Dataset**.

---

## Features

- VGG11 implemented **from scratch in PyTorch**
- Unified **multi-task learning pipeline**
- Custom implementation of:
  - Dropout layer
  - IoU loss for localization
- U-Net style segmentation with skip connections
- Experiment tracking using **Weights & Biases**

---

## Implemented Components

### Classification
- 37-class pet breed classification  
- Fully connected classifier head  

### Localization
- Bounding box regression: *(xc, yc, w, h)*  
- Custom IoU-based evaluation  

### Segmentation
- U-Net style encoder-decoder architecture  
- Pixel-wise trimap prediction  

---

## Evaluation Metrics

| Task           | Metric         |
|----------------|--------------|
| Classification | Macro F1 Score |
| Localization   | IoU           |
| Segmentation   | Dice Score    |

---


## Installation

Clone the repository:

```bash
git clone https://github.com/pmk009/da6401_assignment_2-main.git
cd da6401_assignment_2-main
```

## Acknowledgements

- Claude was used as a conceptual aid for structuring the train.py script  
