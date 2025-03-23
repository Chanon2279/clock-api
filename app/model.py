import torch.nn as nn

class ClockMultiOutput(nn.Module):
    def __init__(self, num_digit_classes=10, num_hand_classes=12):
        super(ClockMultiOutput, self).__init__()
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
        )
        self.digit_out = nn.Linear(128, num_digit_classes)
        self.hand_out = nn.Linear(128, num_hand_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        digit = self.digit_out(x)
        hand = self.hand_out(x)
        return digit, hand
