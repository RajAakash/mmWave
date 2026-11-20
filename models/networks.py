"""
Network architectures for sub-6 GHz → mmWave beam prediction.

- BaselineMLP: reproduces Alrabeiah & Alkhateeb (2020).
- BeamCNN: 1D CNN over subcarriers, treating (real, imag) as 2 channels.
"""
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Fully-connected baseline, matching the architecture style of
    Alrabeiah & Alkhateeb (2020, IEEE TComm).

    Input : flattened [real, imag] sub-6 GHz channel
    Output: logits over mmWave beam codebook
    """
    def __init__(self, input_dim, num_beams=64, hidden_dims=(1024, 512, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_beams))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BeamCNN(nn.Module):
    def __init__(self, input_dim, num_beams=64):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_beams),
        )
    def forward(self, x):
        x = x.unsqueeze(1)   # (B, 1, 151)
        x = self.conv(x)
        return self.head(x)

def build_model(name, input_dim, num_beams):
    if name == 'mlp':
        return BaselineMLP(input_dim, num_beams)
    if name == 'cnn':
        return BeamCNN(input_dim, num_beams)
    raise ValueError(f'Unknown model: {name}')
