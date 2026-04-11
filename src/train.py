import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.data_processing import load_and_clean_clinvar
from src.feature_engineering import extract_features
from src.model import NIVEPNetwork
from src.evaluate import evaluate_model
from src.uncertainty import predict_with_uncertainty
import yaml
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_pipeline(config_path="config.yaml"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_path = config['data']['raw_path']
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(base_dir, raw_path)
        
    df = load_and_clean_clinvar(raw_path)
    
    # Train-test split 
    # TODO: In real biology, split by GENE (GroupKFold) to prevent data leakage!
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    X_train, encoders = extract_features(train_df, fit=True)
    X_test, _ = extract_features(test_df, fit=False, encoders=encoders)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    input_dim = X_train_t.shape[1]
    model = NIVEPNetwork(input_dim, config['model']['hidden_dims'], config['model']['dropout'])
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'])
    
    epochs = config['model']['epochs']
    logger.info(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state_dict = None
    import copy

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            
        if (epoch+1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
    # Load best weights before evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best model with Val Loss: {best_val_loss.item():.4f}")
            
    # Evaluation
    logger.info("Evaluating model...")
    mean_pred, variance = predict_with_uncertainty(model, X_test_t, num_samples=config['model']['mc_samples'])
    metrics = evaluate_model(y_test, mean_pred)
    
    # Save the model and encoders
    logger.info("Saving model and encoders...")
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(base_dir, 'models', 'model.pt'))
    
    with open(os.path.join(base_dir, 'models', 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    with open(os.path.join(base_dir, 'models', 'input_dim.pkl'), 'wb') as f:
        pickle.dump(input_dim, f)
        
    logger.info("Training complete.")
    return model, encoders, metrics

if __name__ == "__main__":
    train_pipeline()
