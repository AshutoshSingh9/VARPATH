import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def explain_predictions(model, X_train_tensor, X_test_tensor):
    """
    Calculates feature attributions. 
    Using Gradient-based attribution as a lightweight alternative to SHAP for PyTorch.
    """
    model.eval()
    X_test_tensor.requires_grad_()
    
    outputs = model(X_test_tensor)
    model.zero_grad()
    
    outputs.sum().backward()
    
    attributions = X_test_tensor.grad.detach().cpu().numpy()
    feature_importance = np.abs(attributions).mean(axis=0)
    return feature_importance
