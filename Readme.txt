Submission details:
The Submission has a folder named Pandi_Sriram_HW1.zip
with
1. a python file named “CNNclassify.py”;
2. a generated model folder named “model”;
3. 4 screenshots 
	1) Training results with Training accuracy obtained while training the model on MNIST dataset(Mnist-Accuracies.png) 
	2) Training results with Training accuracy obtained while training the model on CIFAR dataset(Cifar-Accuracies.png)
	3) Testing Result (testingResult-Mnist.png) for MNIST image-7(Mnist_test_image-1.png)
	4) Testing Result (testingResult-CIFAR.png) for CIFAR image-7(cifar_test_image-1.png)
4. 2 screenshot of the visualization results from the first CONV layer(1 for CIFAR-10 
	output of the first CONV layer in the trained model for each filter (32 visualization results)
	5) CONV_rslt_cifar.png (CONV_rslt_cifar.png)
	6) CONV_rslt_mnist.png (CONV_rslt_mnist.png)

Training: python CNNclassify.py train --cifar
Testing: python CNNclassify.py --dataset cifar test test_image.png 

I have made some changes to my code to accept the dataset as an argument along with the image for testing, so that we can run both mnist and cifar models both through the command line by providing suitable test images for mnist and cifar and predict the class of 2 images from two datasets and display the prediction result, save the visualization results from the first CONV layer of two datasets as  “CONV_rslt_mnist.png” and “CONV_rslt_cifar.png”
