import torch
import torch.nn as nn


class ResBlock(nn.Module):
    # residual block: two convolutions with a skip connection
    # skip connection lets gradients flow directly -> better training on small datasets
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)  # add skip connection
        return x


class SleepApneaCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # block 1: learn local breathing patterns (wide kernel captures one breath ~0.5s)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)  # 960 -> 240

        # residual block at 64 channels
        self.res1 = ResBlock(64, kernel_size=5)

        # block 2: learn multi-breath patterns
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)  # 240 -> 60

        # residual block at 128 channels
        self.res2 = ResBlock(128, kernel_size=3)

        # block 3: high-level event features
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # 60 -> 1

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.res2(x)
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = SleepApneaCNN(n_classes=3)
    dummy = torch.randn(8, 3, 960)
    out = model(dummy)
    print(f"input: {dummy.shape}  output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total:,}")
