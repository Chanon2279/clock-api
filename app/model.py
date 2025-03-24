
import torch.nn as nn

# Model: ClockMultiLabel (Multi-class)
class ClockMultiLabel(nn.Module):
    def __init__(self):
        super(ClockMultiLabel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 4)  # Output 4 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x