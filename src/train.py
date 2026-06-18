import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from src.data_processing import load_and_clean_clinvar
from src.feature_engineering import extract_features
from src.model import NIVEPNetwork
from src.evaluate import evaluate_model
from src.uncertainty import predict_with_uncertainty
import yaml
import logging
import os
import pickle
import random
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _group_key(df: pd.DataFrame) -> np.ndarray:
    """
    Group id = the exact (VariantType, AlleleFrequency, CADD_phred) tuple.
    23% of rows are exact-duplicate feature vectors; a random split scatters
    identical vectors across train and test, so the model memorizes rather than
    generalizes. Grouping on the feature vector keeps every copy on one side of
    the split. (A gene/locus id would be the ideal group key, but the dataset
    does not carry one.)
    """
    key = (df['VariantType'].astype(str) + '|' +
           df['AlleleFrequency'].astype(str) + '|' +
           df['CADD_phred'].astype(str))
    return pd.factorize(key)[0]


def train_pipeline(config_path="config.yaml"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 42)
    set_seed(seed)

    raw_path = config['data']['raw_path']
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(base_dir, raw_path)

    df = load_and_clean_clinvar(raw_path)

    # Group-aware 3-way split (train / val / test, ~60/20/20).
    # The test set is held out and scored exactly once at the end; the val set
    # (carved out of train, never the test set) drives checkpoint selection.
    groups = _group_key(df)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, df['label'], groups))
    df_trainval, df_test = df.iloc[trainval_idx], df.iloc[test_idx]
    groups_tv = groups[trainval_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_trainval, df_trainval['label'], groups_tv))
    train_df, val_df = df_trainval.iloc[train_idx], df_trainval.iloc[val_idx]

    X_train, encoders = extract_features(train_df, fit=True)
    X_val, _ = extract_features(val_df, fit=False, encoders=encoders)
    X_test, _ = extract_features(df_test, fit=False, encoders=encoders)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = df_test['label'].values

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)

    input_dim = X_train_t.shape[1]
    model = NIVEPNetwork(input_dim, config['model']['hidden_dims'], config['model']['dropout'])

    # logits + BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'])

    epochs = config['model']['epochs']
    batch_size = config['model'].get('batch_size') or len(X_train_t)
    logger.info(f"Training for {epochs} epochs (batch_size={batch_size}, seed={seed})...")
    logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(df_test)}")

    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

        # checkpoint selection on the held-out validation set only
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best model with Val Loss: {best_val_loss:.4f}")

    # Headline metric: single deterministic pass on the untouched test set
    # (dropout OFF), so the reported number is reproducible.
    logger.info("Evaluating on held-out test set...")
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(X_test_t)).numpy().ravel()
    metrics = evaluate_model(y_test, test_prob)

    # MC-Dropout uncertainty is reported alongside, not as the headline metric.
    _, variance = predict_with_uncertainty(model, X_test_t, num_samples=config['model']['mc_samples'])
    metrics['mean_epistemic_std'] = float(np.sqrt(variance).mean())

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
