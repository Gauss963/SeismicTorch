"""Three-component waveform classifier."""

import torch
from torch import nn


class EarthquakeCNN(nn.Module):
    """Map [batch, 1, 80, 3] waveforms to P, S and noise logits."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 32, (3, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d((3, 1))
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(32, 64, (3, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d((3, 1))
        self.dropout3 = nn.Dropout(0.1)

        # Three pooling stages leave 2 time steps and 3 components.
        self.fc1 = nn.Linear(64 * 2 * 3, 64)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(64, 16)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(16, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        if x.shape[2] > 2:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (2, 3))
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        return self.fc3(x)


class SmoothedMaxPool(nn.Module):
    """Pool over time, then smooth the responses before decimating."""

    def forward(self, x):
        dense = torch.nn.functional.max_pool2d(x, (3, 1), stride=1)
        neighbors = torch.cat((dense[:, :, :1], dense, dense[:, :, -1:]), dim=2)
        smoothed = (neighbors[:, :, :-2] + 6 * dense + neighbors[:, :, 2:]) / 8
        return smoothed[:, :, ::3]


class SmoothedCNN(EarthquakeCNN):
    """Waveform CNN with smoothed temporal pooling at each stage."""

    def __init__(self):
        super().__init__()
        self.maxpool1 = SmoothedMaxPool()
        self.maxpool2 = SmoothedMaxPool()
        self.maxpool3 = SmoothedMaxPool()


class TemporalBlock(nn.Module):
    """A dilated depthwise filter followed by feature mixing."""

    def __init__(self, width, dilation):
        super().__init__()
        self.filter = nn.Conv1d(width, width, 3, dilation=dilation, padding=dilation,
                                groups=width, bias=False)
        self.mix = nn.Conv1d(width, width, 1, bias=False)
        self.norm = nn.BatchNorm1d(width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = self.norm(self.mix(self.filter(x)))
        return self.activation(x + self.dropout(residual))


class WaveformTCN(nn.Module):
    """A small waveform classifier with early mixing of the three components."""

    def __init__(self, width=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(TemporalBlock(width, d) for d in (1, 2, 4, 8)))
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(nn.Dropout(0.15), nn.Linear(width * 5, 3))

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        features = self.blocks(self.stem(x))
        pooled = torch.cat((self.pool(features).flatten(1), features.amax(dim=-1)), dim=1)
        return self.classifier(pooled)


class PolarizationCNN(EarthquakeCNN):
    """Add rotation-invariant motion and onset contrast to waveform logits."""

    def __init__(self):
        super().__init__()
        self.polarization_head = nn.Linear(9, 3, bias=False)
        nn.init.zeros_(self.polarization_head.weight)

    @staticmethod
    def motion_features(x):
        before, after = x[:, 0].split(x.shape[2] // 2, dim=1)
        features = []
        powers = []
        for samples in (before, after):
            samples = samples - samples.mean(dim=1, keepdim=True)
            covariance = samples.transpose(1, 2) @ samples / samples.shape[1]
            total = covariance.diagonal(dim1=1, dim2=2).sum(dim=1)
            power = total.clamp_min(1e-8)
            # The trace and Frobenius norm are invariant under rotation.
            concentration = covariance.square().sum(dim=(1, 2)) / power.square()
            vertical = covariance[:, 2, 2] / power
            features.extend((vertical, concentration, power.sqrt()))
            powers.append(power)
        features.extend(((powers[1] / powers[0]).log().clamp(-4, 4) / 4,
                         features[3] - features[0], features[4] - features[1]))
        return torch.stack(features, dim=1)

    def forward(self, x):
        return super().forward(x) + self.polarization_head(self.motion_features(x))


WAVEFORM_MODELS = {
    "cnn": EarthquakeCNN,
    "tcn": WaveformTCN,
    "polarization": PolarizationCNN,
    "smoothed": SmoothedCNN,
}


def initialize_waveform_model(name, weights):
    model = WAVEFORM_MODELS[name]()
    # Starting a polarization experiment from the CNN preserves its logits.
    if name == "polarization" and "polarization_head.weight" not in weights:
        weights = {**weights, "polarization_head.weight": model.polarization_head.weight.detach()}
    model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    model = EarthquakeCNN()
    print(model(torch.zeros(1, 1, 80, 3)).shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
