import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, mobilenet_v2, resnet18, vgg11
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("saved_models", exist_ok=True)

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
            product_spec_norms *= spectral
            col_norms = torch.norm(diff, p=2, dim=0)
            norm_21 = torch.sum(col_norms).item()
            sum_mixed_norms += (norm_21 ** (2 / 3))

    return product_spec_norms, sum_mixed_norms

# Certificate function
def certificate(sample_size: int, gamma: float, model: nn.Module):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=sample_size, shuffle=True)
    inputs, _ = next(iter(loader))
    X = inputs.view(sample_size, -1).to(device)

    X_norm = torch.linalg.norm(X, ord=2).item()
    spec_complexity, _ = spectral_complexity(model)

    max_out_dim = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
            out_dim = module.weight.data.shape[0]
            max_out_dim = max(max_out_dim, out_dim)

    return (max_out_dim * X_norm * spec_complexity) / (sample_size * gamma)

# Evaluation and plotting with certificate
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

subset_size = int(0.01 * len(testset))
subset_indices = random.sample(range(len(testset)), subset_size)
test_subset = Subset(testset, subset_indices)

num_classes = 10

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Load saved ResNet18 model
checkpoint_path = "saved_models/ResNet18.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✅ Loaded saved ResNet18 model from disk.")
else:
    raise FileNotFoundError("❌ Model checkpoint not found. Please train and save the model first.")

model.eval()


# Compute gamma from score differences
def get_top2_differences(model, dataset, device):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    diffs, correct, total = [], 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            top2 = torch.topk(outputs, k=2, dim=1).values
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            diffs.extend((top2[:, 0] - top2[:, 1]).cpu().numpy())
    acc = correct / total
    return diffs, acc

train_diffs, train_acc = get_top2_differences(model, trainset, device)
test_diffs, test_acc = get_top2_differences(model, test_subset, device)
true_gamma = min(test_diffs)

# Compute certificate
cert_value = certificate(sample_size=len(test_subset), gamma=true_gamma, model=model)

# Plot
plt.figure(figsize=(10, 5))
plt.hist(train_diffs, bins=50, alpha=0.6, label="Train Gamma", color="blue", density=True)
plt.hist(test_diffs, bins=50, alpha=0.6, label="Test Gamma (1%)", color="orange", density=True)
plt.xlabel("Score Difference (Top1 - Top2)")
plt.ylabel("Density")
plt.title(f"ResNet18 - Gamma Distribution\nTrain Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%} | Certificate: {cert_value:.4f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gamma_hist_ResNet18.png")
plt.show()
