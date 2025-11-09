import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, mobilenet_v2, resnet18, vgg11
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LeNet5
class LeNet5(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model by name and weight path
def load_model(model_name: str, weight_path: str, num_classes=10):
    if model_name == "LeNet5":
        model = LeNet5(in_channels=3, num_classes=num_classes)
    elif model_name == "AlexNet":
        model = alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "MobileNetV2":
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet18":
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG11":
        model = vgg11(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Spectral complexity function
def spectral_complexity(model: nn.Module):
    product_spec_norms = 1.0
    sum_mixed_norms = 0.0

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
            weight = module.weight.data
            weight_2d = weight.view(weight.size(0), -1)
            weight_t = weight_2d.t()
            spectral = torch.linalg.norm(weight_2d, ord=2).item()
            rows, cols = weight_t.shape
            identity = torch.zeros_like(weight_t)
            for i in range(min(rows, cols)):
                identity[i, i] = 1.0

            diff = weight_t - identity

            # Spectral norm
            
            product_spec_norms *= spectral

            # (2,1)-norm ^ (2/3)
            col_norms = torch.norm(diff, p=2, dim=0)
            norm_21 = torch.sum(col_norms).item()
            sum_mixed_norms += (norm_21 ** (2 / 3)) / (spectral ** (2 / 3))

    return product_spec_norms * (sum_mixed_norms ** (3 / 2))
