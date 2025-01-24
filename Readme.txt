# **CNN Architecture Implementation for MNIST and CIFAR-10**

This project implements a **Convolutional Neural Network (CNN)** for classifying images from the MNIST and CIFAR-10 datasets. The implementation is done in Python using the **PyTorch** framework. It includes training, testing, and visualizing the results of the first convolutional layer.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Usage Instructions](#usage-instructions)
   - [Training the Model](#training-the-model)
   - [Testing the Model](#testing-the-model)
5. [Results](#results)
   - [Training Accuracy](#training-accuracy)
   - [Testing Accuracy](#testing-accuracy)
   - [First Convolutional Layer Visualization](#first-convolutional-layer-visualization)
6. [Implementation Details](#implementation-details)
7. [File Structure](#file-structure)
8. [Acknowledgements](#acknowledgements)

---

## **Project Overview**
This project aims to:
- Implement a CNN classifier from scratch for the **MNIST** and **CIFAR-10** datasets.
- Understand the architecture and working of CNNs, including **backpropagation** and **gradient descent**.
- Optimize the model to achieve a **testing accuracy of ≥ 75% on CIFAR-10**.
- Visualize the filters from the first convolutional layer of the trained CNN.

---

## **Features**
- Implements **training** and **testing** functions for CNNs.
- Saves trained models to a `model/` directory.
- Visualizes the outputs of the first convolutional layer for both datasets.
- Achieves high testing accuracy with optimization techniques:
  - **Mini-batch gradient descent**
  - **Batch normalization**
  - **Dropout**
  - **Regularization**

---

## **Dependencies**
Ensure you have the following Python packages installed:
- `torch`
- `torchvision`
- `numpy`
- `opencv-python`

Install them using pip:
```bash
pip install torch torchvision numpy opencv-python
