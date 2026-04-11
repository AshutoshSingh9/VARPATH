import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import extract_features

def test_feature_engineering():
    df = pd.DataFrame({
        'VariantType': ['missense', 'nonsense'],
        'AlleleFrequency': ['0.01', '0.005']
    })
    
    X, encoders = extract_features(df, fit=True)
    assert X.shape[0] == 2
    assert X.shape[1] > 0
    assert 'var_type' in encoders
