import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_clean_clinvar(filepath: str) -> pd.DataFrame:
    """
    Loads ClinVar dataset and cleans labels for binary classification.
    Leaves missing feature handling (NaNs) to feature_engineering to prevent leakage.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise

    assert 'ClinicalSignificance' in df.columns, "Missing ClinicalSignificance column"

    label_map = {
        'Benign': 0,
        'Likely benign': 0,
        'Benign/Likely benign': 0,
        'Pathogenic': 1,
        'Likely pathogenic': 1,
        'Pathogenic/Likely pathogenic': 1
    }
    
    # We might have case differences or multiple tags. To be safe, just title-case.
    # The API might return lowercase like 'pathogenic' or 'benign'.
    df['ClinicalSignificance'] = df['ClinicalSignificance'].astype(str).str.capitalize()
    
    # Re-map standardizing just in case 
    label_map_extended = {
        'Benign': 0,
        'Likely benign': 0,
        'Benign/likely benign': 0,
        'Pathogenic': 1,
        'Likely pathogenic': 1,
        'Pathogenic/likely pathogenic': 1
    }
    
    df['label'] = df['ClinicalSignificance'].map(label_map_extended)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)

    logger.info(f"Loaded {len(df)} variants after cleaning.")
    return df
