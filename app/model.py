import torch
import torch.nn as nn

class ClockMultiLabel(nn.Module):
    def __init__(self):
        super(ClockMultiLabel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
