import torch
import torch.nn as nn

class NIVEPNetwork(nn.Module):
    """
    MLP with Monte Carlo Dropout for uncertainty estimation.
    forward() returns raw logits; apply sigmoid at loss/inference time
    so training can use the numerically stable BCEWithLogitsLoss.
    """
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))
