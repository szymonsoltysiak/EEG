# EEGNet
import torch
import torch.nn as nn

class EEGClassifier(nn.Module):
    """
    EEGNet implementation (Lawhern et al., 2018)
    Adapted for 1D EEG input: (batch, channels, time)
    """
    def __init__(self, num_channels=16, num_classes=4, time_points=128):
        super(EEGClassifier, self).__init__()

        # --- Parameters ---
        F1 = 8     # number of temporal filters
        D = 2      # depth multiplier (spatial filters per temporal filter)
        F2 = F1 * D
        kernel_length = 64

        # --- Block 1: Temporal Convolution ---
        self.conv_temporal = nn.Conv2d(
            1, F1,
            kernel_size=(1, kernel_length),
            stride=1,
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # --- Block 2: Depthwise Spatial Convolution ---
        self.conv_spatial = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(num_channels, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.25)

        # --- Block 3: Separable Convolution ---
        self.conv_sep_depth = nn.Conv2d(
            F1 * D, F1 * D,
            kernel_size=(1, 16),
            groups=F1 * D,
            padding=(0, 8),
            bias=False
        )
        self.conv_sep_point = nn.Conv2d(
            F1 * D, F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(0.25)

        # --- Final classification layer ---
        # Compute feature dimension after convs
        dummy_input = torch.zeros(1, 1, num_channels, time_points)
        with torch.no_grad():
            out = self._forward_features(dummy_input)
            self.flatten_dim = out.shape[1]
        self.fc = nn.Linear(self.flatten_dim, num_classes)

    def _forward_features(self, x):
        x = self.conv_temporal(x)
        x = self.bn1(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv_sep_depth(x)
        x = self.conv_sep_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = x.unsqueeze(1)  # -> (batch, 1, channels, time)
        x = self._forward_features(x)
        x = self.fc(x)
        return x

# Hyperparameters
params = {
    'batch_size': 512,
    'learning_rate': 0.01,
    'num_epochs': 50,
    'num_classes': 4,
    'window_size': 128,
    'num_channels': 16,
    'step_size': 10,
    'gamma': 0.5
}
