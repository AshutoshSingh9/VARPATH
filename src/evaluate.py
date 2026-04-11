import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred_prob, threshold=0.5):
    """
    Evaluates the model and logs metrics.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float('nan') # Handle cases with only one class in test set
        
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }
    
    logger.info(f"Evaluation Metrics: Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return metrics
