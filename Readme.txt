# CNN Architecture Implementation for MNIST and CIFAR-10

This project implements a **Convolutional Neural Network (CNN)** for classifying images from the MNIST and CIFAR-10 datasets. The implementation is done in Python using the **PyTorch** framework. It includes training, testing, and visualizing the results of the first convolutional layer.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Usage Instructions](#usage-instructions)
5. [Results](#results)
6. [Implementation Details](#implementation-details)
7. [File Structure](#file-structure)
8. [Acknowledgements](#acknowledgements)

---

## Project Overview
This project aims to:
- Implement a CNN classifier from scratch for the **MNIST** and **CIFAR-10** datasets.
- Understand the architecture and working of CNNs, including **backpropagation** and **gradient descent**.
- Achieve a **testing accuracy of ≥ 75% on CIFAR-10**.
- Visualize the filters from the first convolutional layer of the trained CNN.

---

## Features
- Implements **training** and **testing** functions for CNNs.
- Saves trained models in a `model/` directory.
- Visualizes outputs of the first convolutional layer for both datasets.
- Uses optimization techniques:
  - Mini-batch gradient descent
  - Batch normalization
  - Dropout
  - Regularization

---

## Dependencies
Install the required Python packages using pip:
```bash
pip install torch torchvision numpy opencv-python


## Usage Instructions
Training the Model For MNIST:
python CNNclassify.py train --mnist

For CIFAR-10:
python CNNclassify.py train --cifar

## Testing the Model
Provide an image file (e.g., xxx.png) from the dataset.
python CNNclassify.py test xxx.png

## Outputs:
Predicts the class of the image and displays it.
Saves visualizations of the first convolutional layer as:
CONV_rslt_mnist.png
CONV_rslt_cifar.png

## Results
Training Accuracy
MNIST: ~99%
CIFAR-10: ~75%
Testing Accuracy
MNIST: ~99%
CIFAR-10: ≥ 75%

First Convolutional Layer Visualization
Visualizations highlight the feature extraction capability of the filters.

## Implementation Details
Datasets:
MNIST: Grayscale images.
CIFAR-10: RGB images across 10 classes.

## Network Architecture:
First convolutional layer:
Filter size: 5x5
Stride: 1
Filters: 32

## Additional layers:
Multiple CONV and FC layers with dropout and batch normalization.

## Optimization:
Mini-batch gradient descent
Batch normalization
Dropout

## File Structure
.
├── CNNclassify.py         # Main script for training and testing
├── model/                 # Directory for trained models
├── results/               # Directory for visualizations
└── README.md              # This file

## Acknowledgements
This project was part of EECE 7398 - Advances in Deep Learning (Fall 2023) at Northeastern University.
