import torch
import numpy as np

def predict_with_uncertainty(model, X_tensor, num_samples=20):
    """
    Monte Carlo Dropout for epistemic uncertainty only.
    Dropout is kept ON (model.train()) on purpose here so each pass is a
    different sub-network. This is NOT used for the headline metric, which is
    a single deterministic eval() pass (see train.py) so it stays reproducible.
    The model returns logits, so we squash with sigmoid to get probabilities.
    """
    model.train()
    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            pred = torch.sigmoid(model(X_tensor))
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)  # epistemic uncertainty

    return mean_pred, variance
