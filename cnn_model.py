import torch
import torch.nn as nn
import torch.nn.functional as F

class FloodCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Final feature map = 64 × 16 × 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)

        # 2 classes → Flooded / Non-Flooded
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128→64
        x = self.pool(F.relu(self.conv2(x)))  # 64 →32
        x = self.pool(F.relu(self.conv3(x)))  # 32→16

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)   # raw logits (NO SIGMOID)
