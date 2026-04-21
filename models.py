import torch.nn as nn
import torch

class CNN_Baseline(nn.Module):
    """A simple CNN for 2D Mel Spectrogram classification."""

    def __init__(self, input_shape, output_dim):
        super(CNN_Baseline, self).__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(C, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(64),
        )

        # Compute flattened size after conv blocks
        dummy = torch.zeros(1, C, H, W)
        flat_dim = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN_Advanced(nn.Module):
    """A more advanced CNN for 2D Mel Spectrogram classification."""

    def __init__(self, input_shape, output_dim):
        super(CNN_Advanced, self).__init__()
        C, H, W = input_shape

        ### <--- START OF YOUR CODE

        self.features = nn.Sequential(
            # First Layer
            nn.Conv2d(C, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(8),
            
            # Second Layer
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(8),
        )

        # Compute flattened size after conv blocks
        dummy = torch.zeros(1, C, H, W)
        flat_dim = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        ### END OF YOUR CODE --->

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNN_Improved(nn.Module):
    """Improved CNN for 2D Mel Spectrogram classification."""

    def __init__(self, input_shape, output_dim):
        super(CNN_Improved, self).__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

        # Global Average Pooling collapses (H, W) → (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def loss(device):
    """ Returns the loss function for the baseline MLP model.
    """
    loss = nn.CrossEntropyLoss()
    return loss


