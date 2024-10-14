import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from torchvision import models


# Modify the VGG16 model
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=10, channel=3):
        super(VGG16Classifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        # Modify the first convolutional layer to accept 3-channel input if needed
        # self.vgg16.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # Modify the classifier to fit the number of classes in your dataset
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
        return x


# VGG-like model
class VGG(nn.Module):
    def __init__(self, num_classes, channel):
        super(VGG, self).__init__()
        self.features = nn.Sequential(OrderedDict([

            # Block 1
            ('block1_conv1', nn.Conv2d(channel, 256, kernel_size=3, padding=1)),
            ('block1_relu1', nn.ReLU(inplace=True)),
            ('block1_conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('block1_relu2', nn.ReLU(inplace=True)),
            ('block1_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),

            # Block 2
            ('block2_conv1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ('block2_relu1', nn.ReLU(inplace=True)),
            ('block2_conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block2_relu2', nn.ReLU(inplace=True)),
            ('block2_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),

            # # Block 3
            # ('block3_conv1', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            # ('block3_relu1', nn.ReLU(inplace=True)),
            # ('block3_conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            # ('block3_relu2', nn.ReLU(inplace=True)),
            # ('block3_pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
        ]))

        # Adjusted classifier for 32x32x512 output after Block 3
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 32 * 32, 4096),  # Adjusted for 32x32 output
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    #
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     return x


# VGG-like model with named layers for easy reference
class VGGOriginal(nn.Module):
    def __init__(self, num_classes, channel):
        super(VGGOriginal, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            # Conv Block 1
            ('block1_conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('block1_relu1', nn.ReLU(inplace=True)),
            ('block1_conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('block1_relu2', nn.ReLU(inplace=True)),
            ('block1_pool', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Block 2
            ('block2_conv1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('block2_relu1', nn.ReLU(inplace=True)),
            ('block2_conv2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('block2_relu2', nn.ReLU(inplace=True)),
            ('block2_pool', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Block 3
            ('block3_conv1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('block3_relu1', nn.ReLU(inplace=True)),
            ('block3_conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('block3_relu2', nn.ReLU(inplace=True)),
            ('block3_conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('block3_relu3', nn.ReLU(inplace=True)),
            ('block3_pool', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Block 4
            ('block4_conv1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ('block4_relu1', nn.ReLU(inplace=True)),
            ('block4_conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block4_relu2', nn.ReLU(inplace=True)),
            ('block4_conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block4_relu3', nn.ReLU(inplace=True)),
            ('block4_pool', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Block 5
            ('block5_conv1', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block5_relu1', nn.ReLU(inplace=True)),
            ('block5_conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block5_relu2', nn.ReLU(inplace=True)),
            ('block5_conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('block5_relu3', nn.ReLU(inplace=True)),
            ('block5_pool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
