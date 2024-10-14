import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class Classifier(nn.Module):
    def __init__(self, num_classes, channel):
        super(Classifier, self).__init__()

        # Adjust kernel sizes and padding based on your data complexity and requirements
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Adjust stride and kernel size if needed

        # Consider adding a second convolutional layer for better feature extraction
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Example second layer

        # Calculate the output size after convolutional and pooling layers:
        n = 1  # Assuming a batch size of 1 for this example
        c = 32  # Number of channels after conv2 (adjust if needed)  # Assuming two conv layers
        w = 2  # Width after two pooling layers (adjust based on conv layers)
        h = 2  # Height after two pooling layers (adjust based on conv layers)
        flattened_size = n * c * w * h

        self.fc1 = nn.Linear(128, 128)  # Adjust hidden units as needed
        self.fc2 = nn.Linear(128, num_classes)  # Output layer with num_classes units

    def forward(self, x):  # Ignore 'sleep' input if not used
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv1(torch.reshape(x, (64, 3, 64, 64)))))

        # Consider adding a second convolutional layer for better feature extraction
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sleep(self, x):
        return x * torch.randn(x.shape)

    def expand_fc_layer(self, additional_neurons):
        # Expand the FC layer by adding more neurons
        current_weight = self.fc.weight.data
        current_bias = self.fc.bias.data
        new_weight = torch.cat([current_weight, torch.zeros(additional_neurons, current_weight.shape[1])], dim=0)
        new_bias = torch.cat([current_bias, torch.zeros(additional_neurons)], dim=0)
        self.fc3 = nn.Linear(self.fc.in_features, self.fc.out_features + additional_neurons)
        self.fc3.weight.data = new_weight
        self.fc3.bias.data = new_bias
        self.output_neurons += additional_neurons
