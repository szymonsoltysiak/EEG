# EEGBase
import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(EEGClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(x).squeeze(-1)  # shape: (batch, 128)
        x = self.fc(x)                # shape: (batch, num_classes)
        return x

# Hyperparameters
params = {
    'batch_size': 512,
    'learning_rate': 0.01,
    'num_epochs': 50,
    'num_classes': 4,
    'window_size': 128,
    'num_channels': 16,
    'step_size': 20,
    'gamma': 0.5
}