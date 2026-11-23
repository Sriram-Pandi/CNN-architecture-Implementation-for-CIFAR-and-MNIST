# CNN Image Classification on MNIST and CIFAR-10

[CNNclassify.py](CNNclassify.py) trains and tests a PyTorch convolutional network. Install PyTorch, torchvision, OpenCV, and Matplotlib, then run from the repository root.

```bash
python CNNclassify.py train --dataset mnist
python CNNclassify.py train --dataset cifar
python CNNclassify.py test --dataset mnist Mnist_test_image-1.png
python CNNclassify.py test --dataset cifar cifar_test_image-1.png
```

Testing loads the corresponding checkpoint from `model/`, prints the predicted class, and saves a first-layer feature-map figure. The repository also includes training accuracy plots and saved test outputs.
