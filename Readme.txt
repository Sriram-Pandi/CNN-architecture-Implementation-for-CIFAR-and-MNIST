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
```

---

## **Usage Instructions**

### **Training the Model**
To train the model, use the following commands:
- For MNIST:
  ```bash
  python CNNclassify.py train --mnist
  ```
- For CIFAR-10:
  ```bash
  python CNNclassify.py train --cifar
  ```

The trained models will be saved in the `model/` directory.

---

### **Testing the Model**
To test the model and visualize results:
1. Provide an image file (`xxx.png`) from the respective dataset.
2. Run the following command:
   ```bash
   python CNNclassify.py test xxx.png
   ```
3. This will:
   - Predict the class of the image and display the result.
   - Save the visualization of the first convolutional layer's filters as:
     - `CONV_rslt_mnist.png` (for MNIST)
     - `CONV_rslt_cifar.png` (for CIFAR-10)

---

## **Results**

### **Training Accuracy**
- **MNIST**: Achieved accuracy: `~99%`
- **CIFAR-10**: Achieved accuracy: `~75%`

### **Testing Accuracy**
- **MNIST**: Final testing accuracy: `~99%`
- **CIFAR-10**: Final testing accuracy: `≥ 75%`

### **First Convolutional Layer Visualization**
- Visualizations show the filters' response to the input images, highlighting feature extraction at an early stage of the network.

---

## **Implementation Details**
1. **Dataset**:
   - MNIST: Grayscale digit images.
   - CIFAR-10: RGB images across 10 classes.
2. **Network Architecture**:
   - First Convolutional Layer:
     - Filter size: `5x5`
     - Stride: `1`
     - Filters: `32`
   - Additional Layers:
     - No restrictions, multiple CONV and FC layers used.
   - Optimization:
     - Mini-batch gradient descent
     - Batch normalization
     - Dropout
3. **Training**:
   - Epochs: Configurable
   - Loss Function: CrossEntropyLoss
   - Optimizer: SGD with momentum

---

## **File Structure**
```
.
├── CNNclassify.py         # Main script for training and testing
├── model/                 # Directory for saving trained models
├── results/               # Directory for saving visualization results
└── README.md              # This file
```

---

## **Acknowledgements**
This project was implemented as part of **EECE 7398 - Advances in Deep Learning (Fall 2023)** at Northeastern University.
