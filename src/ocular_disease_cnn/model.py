import torch
import torch.nn as nn
import torch.nn.functional as F

class OcularCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Bloques convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Clasificador denso
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # suponiendo entrada 224x224
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
