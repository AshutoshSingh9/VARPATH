import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def explain_predictions(model, X_test_tensor):
    """
    Gradient-based feature attribution (lightweight alternative to SHAP).
    Works on a detached clone so the caller's tensor is never mutated.
    The model returns logits, so attributions are gradients of the logit.
    """
    model.eval()
    x = X_test_tensor.detach().clone().requires_grad_(True)

    outputs = model(x)
    model.zero_grad()
    outputs.sum().backward()

    attributions = x.grad.detach().cpu().numpy()
    feature_importance = np.abs(attributions).mean(axis=0)
    return feature_importance
