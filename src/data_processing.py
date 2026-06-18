import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical ClinVar significance strings -> binary label.
# Keys are casefolded so matching is case-insensitive without corrupting
# multi-token labels the way str.capitalize() did.
LABEL_MAP = {
    'benign': 0,
    'likely benign': 0,
    'benign/likely benign': 0,
    'pathogenic': 1,
    'likely pathogenic': 1,
    'pathogenic/likely pathogenic': 1,
}

def load_and_clean_clinvar(filepath: str) -> pd.DataFrame:
    """
    Loads ClinVar dataset and cleans labels for binary classification.
    Unmapped significance values (e.g. 'uncertain significance',
    'conflicting interpretations') are dropped explicitly and logged, so
    silent data loss is visible. Missing feature handling (NaNs) is left to
    feature_engineering to keep imputation fit on the training split only.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

    assert 'ClinicalSignificance' in df.columns, "Missing ClinicalSignificance column"

    sig = df['ClinicalSignificance'].astype(str).str.strip().str.casefold()
    df['label'] = sig.map(LABEL_MAP)

    unmapped = df['label'].isna()
    if unmapped.any():
        breakdown = sig[unmapped].value_counts().head(10).to_dict()
        logger.info(f"Dropping {int(unmapped.sum())} rows with unmapped significance: {breakdown}")

    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)

    logger.info(f"Loaded {len(df)} variants after cleaning.")
    return df
