from torch import nn
import torch


class MusicClassifier(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(
                in_features=input_features, out_features=2048, dtype=torch.float32
            ),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=2048, out_features=1024, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=1024, out_features=512, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=512, out_features=256, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=256, out_features=128, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=128, out_features=64, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(
                in_features=64, out_features=output_features, dtype=torch.float32
            ),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
