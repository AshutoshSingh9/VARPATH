import torch
import torch.nn as nn

class NIVEPNetwork(nn.Module):
    """
    Neural Network with Monte Carlo Dropout for uncertainty estimation.
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
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return torch.sigmoid(logits)
