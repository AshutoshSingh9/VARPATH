import torch
import numpy as np

def predict_with_uncertainty(model, X_tensor, num_samples=20):
    """
    Monte Carlo Dropout to estimate uncertainty.
    model.train() is kept active to ensure dropout is applied at inference.
    """
    model.train() 
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_tensor)
            predictions.append(pred.cpu().numpy())
            
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0) # Epistemic uncertainty
    
    return mean_pred, variance
