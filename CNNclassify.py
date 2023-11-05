import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import cv2
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# Model Definition
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.input_channels = input_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
# Due to bad accuracies with CIFAR dataset, Had to create another layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
#For Cifar and Mnist since the dataset has different type of images the number of channels is different 
        if self.input_channels == 3:  # CIFAR-10
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
        else:  # MNIST
            self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if self.input_channels == 3:  # CIFAR-10
            out = out.view(out.size(0), 64 * 8 * 8)
        else:  # MNIST
            out = out.view(out.size(0), 64 * 7 * 7)  # Reshape for MNIST
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



# Training the model 
def train_model(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "mnist":
        model = CNN(input_channels=1).to(device)
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test)

    elif dataset_name == "cifar":
        model = CNN(input_channels=3).to(device)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Had to add these for increasing the accuracy of of Model while training it on CIFAR dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

    for epoch in range(10):  
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # since asked for only Test accuracy Printing test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/10], Test accuracy: {accuracy:.2f}%")

    # Saving the model
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(model.state_dict(), f"model/{dataset_name}_model.pth")

# Testing Function
# added the datset_name argument as well to make it easy whhile testing any image of either mnist or cifar 
#just providing the dataset name along with the image path it gives better prediction results

def test_image(image_path, dataset_name):
    # Defining the transformations
    if dataset_name == "cifar":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),  # Resize to 32x32 for CIFAR
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_channels = 3
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #For some reason I was getting errors....so had to declare these classes 
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),  # Resize to 28x28 for MNIST
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        input_channels = 1
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        classes = [str(i) for i in range(10)]  # MNIST has classes 0-9

    
    # Transforming the image
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the appropriate models
    model = CNN(input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(f"model/{dataset_name}_model.pth"))

    with torch.no_grad():
        model.eval()
        outputs = model(image.to(device))
        _, predicted = outputs.max(1)
        print(f"Predicted class: {classes[predicted.item()]}")

        # To Visualize the output of the first convolutional layer
        first_conv_layer = model.layer1[0]  # Accessing the first conv layer in 'layer1' Sequential block
        first_layer_output = first_conv_layer(image.to(device))

        feature_maps = first_layer_output.cpu().detach()

        # Create a figure to hold the subplots for each filter's feature map
        fig, axs = plt.subplots(4, 8, figsize=(15, 8))  # 32 filters in a 4x8 grid
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()

        for i in range(feature_maps.size(1)):  # feature_maps.size(1) should give the number of filters
            feature_map = feature_maps[0, i, :, :]  # Selecting one feature map

            # Normalize to [0,1] range for display
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

            # Plotting
            axs[i].imshow(feature_map, cmap='gray')
            axs[i].set_title(f'Filter {i+1}')
            axs[i].axis('off')

        plt.savefig(f"CONV_rslt_{dataset_name}.png")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["train", "test"])
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar"], required=True)
    parser.add_argument("image_path", type=str, nargs="?")
    args = parser.parse_args()

    if args.action == "train":
        train_model(args.dataset)
    elif args.action == "test" and args.image_path:
        test_image(args.image_path, args.dataset)
