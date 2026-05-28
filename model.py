import torch
import torch.nn as nn


class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()

        # first convolutional layer - learns basic features like edges and colours
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)

        # second convolutional layer - learns more complex shapes from the first layer's output
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)

        # third convolutional layer - learns higher level features like fur patterns or beaks
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3   = nn.BatchNorm2d(128)

        # max pooling halves the spatial size after each of the first two blocks
        self.pool = nn.MaxPool2d(2, 2)

        # adaptive pooling collapses whatever size we have down to 1x1
        # this means the linear layers always get 128 values no matter the input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # first fully connected layer - combines all the features
        self.fc1 = nn.Linear(128, 256)

        # dropout randomly turns off neurons during training
        # this stops the model from just memorising the training data
        self.dropout = nn.Dropout(0.5)

        # final layer - outputs 3 values, one for each animal class
        # (bird=0, cat=1, dog=2)
        self.fc2 = nn.Linear(256, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        # first block: conv -> batchnorm -> relu -> maxpool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # second block: conv -> batchnorm -> relu -> maxpool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # third block: conv -> batchnorm -> relu (no pooling here)
        x = self.relu(self.bn3(self.conv3(x)))

        # squeeze the spatial dimensions down to 1x1
        x = self.adaptive_pool(x)

        # flatten from (batch, 128, 1, 1) to (batch, 128)
        x = x.view(x.size(0), -1)

        # fully connected layers with dropout in between
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
