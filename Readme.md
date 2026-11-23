# CNN Image Classification on MNIST and CIFAR-10

[CNNclassify.py](CNNclassify.py) trains and tests a PyTorch convolutional network. Install PyTorch, torchvision, OpenCV, and Matplotlib, then run from the repository root.

```bash
python CNNclassify.py train --dataset mnist
python CNNclassify.py train --dataset cifar
python CNNclassify.py test --dataset mnist Mnist_test_image-1.png
python CNNclassify.py test --dataset cifar cifar_test_image-1.png
```

Testing loads the corresponding checkpoint from `model/`, prints the predicted class, and saves a first-layer feature-map figure. The repository also includes training accuracy plots and saved test outputs.

## Inputs and outputs

Training downloads the selected torchvision dataset and writes a checkpoint to `model/`. Test mode expects a readable image, uses the matching checkpoint, prints a label, and writes `CONV_rslt_mnist.png` or `CONV_rslt_cifar.png`. Run the command examples from the repository root. Start with a checked-in test image and checkpoint before retraining so you can confirm inference and plot generation independently.
