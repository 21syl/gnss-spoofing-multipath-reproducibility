"""Neural architectures specified in manuscript Table 3 and Section IV.

Inputs are [batch, correlators, time]; outputs are unnormalized class logits.
Baseline configurations follow manuscript Table 3.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    """Three k=3 Conv1D/BN/ReLU/MaxPool stages, followed by a classifier."""

    def __init__(self, channels=11, classes=3):
        super().__init__()
        layers = []
        for width in (64, 128, 256):
            layers.extend([nn.Conv1d(channels, width, 3, padding=1),
                           nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)])
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class LSTM(nn.Module):
    """Two unidirectional LSTM layers with 128 hidden units per layer."""

    def __init__(self, channels=11, classes=3):
        super().__init__()
        self.recurrent = nn.LSTM(channels, 128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        sequence, _ = self.recurrent(x.transpose(1, 2))
        return self.classifier(sequence[:, -1])


class ResidualBlock(nn.Module):
    """Two k=3 convolutions; a pointwise projection when channel width changes."""

    def __init__(self, incoming, outgoing):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 3, padding=1), nn.BatchNorm1d(outgoing),
            nn.ReLU(), nn.Conv1d(outgoing, outgoing, 3, padding=1),
            nn.BatchNorm1d(outgoing))
        self.skip = (nn.Identity() if incoming == outgoing else
                     nn.Conv1d(incoming, outgoing, 1))

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Four residual blocks with widths 64/128/256/512."""

    def __init__(self, channels=11, classes=3):
        super().__init__()
        blocks = []
        for width in (64, 128, 256, 512):
            blocks.append(ResidualBlock(channels, width))
            channels = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(512, classes)

    def forward(self, x):
        return self.classifier(self.blocks(x).mean(-1))


class PositionalEncoding(nn.Module):
    def __init__(self, width=256, max_length=1000):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(max_length, width)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError("Sequence exceeds positional-encoding capacity")
        return x + self.encoding[:, :x.size(1)]


class Transformer(nn.Module):
    """Four encoder layers, d=256, 8 heads, FFN=512, sinusoidal positions."""

    def __init__(self, channels=11, classes=3):
        super().__init__()
        self.project = nn.Linear(channels, 256)
        self.position = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(256, classes)

    def forward(self, x):
        x = self.position(self.project(x.transpose(1, 2)))
        return self.classifier(self.encoder(x).mean(1))


class Proposed(nn.Module):
    """Spatial encoding, multiscale branches, two-layer BiLSTM and self-attention.

    Adapted from FullLSTM in train_mixed_order_transfer_20261020.py.
    Feature channels are mixed by the 1x1 convolution; Conv1D kernels operate
    along time. The second encoding kernel is 3, as specified in Section IV.
    """

    def __init__(self, channels=11, classes=3, spatial=True, multiscale=True):
        super().__init__()
        self.spatial = (nn.Sequential(
            nn.Conv1d(channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
            if spatial else nn.Identity())
        width = 128 if spatial else channels
        self.multiscale = multiscale
        if multiscale:
            self.branches = nn.ModuleList([
                nn.Sequential(nn.MaxPool1d(4), nn.Conv1d(width, 64, 15, padding=7),
                              nn.BatchNorm1d(64), nn.ReLU()),
                nn.Sequential(nn.MaxPool1d(2), nn.Conv1d(width, 64, 7, padding=3),
                              nn.BatchNorm1d(64), nn.ReLU()),
                nn.Sequential(nn.Conv1d(width, 64, 3, padding=1),
                              nn.BatchNorm1d(64), nn.ReLU())])
            width = 192
        self.recurrent = nn.LSTM(width, 256, num_layers=2,
                                 bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(512, 8, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, x):
        x = self.spatial(x)
        if self.multiscale:
            length = x.size(-1)
            x = torch.cat([F.interpolate(branch(x), size=length, mode="linear",
                                        align_corners=False) for branch in self.branches], 1)
        x, _ = self.recurrent(x.transpose(1, 2))
        attended, _ = self.attention(x, x, x, need_weights=False)
        return self.classifier(self.norm(x + attended).mean(1))


MODELS = {"cnn": CNN, "lstm": LSTM, "resnet1d": ResNet1D,
          "transformer": Transformer, "proposed": Proposed}


def build_model(name, channels=11, classes=3):
    ablations = {"no_spatial": (False, True), "no_multiscale": (True, False),
                 "no_spatial_no_multiscale": (False, False)}
    if name in ablations:
        spatial, multiscale = ablations[name]
        return Proposed(channels, classes, spatial, multiscale)
    return MODELS[name](channels, classes)
