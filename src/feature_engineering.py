import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def extract_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    Extracts tabular and biological features for the ML model.
    Handles imputations correctly to prevent data leakage.
    """
    if encoders is None:
        encoders = {}
    features = []
    
    # 1. Variant Type (Tabular / Categorical)
    if 'VariantType' in df.columns:
        # Fill missing variant types with 'Unknown'
        var_types = df[['VariantType']].fillna('Unknown')
        if fit:
            encoders['var_type'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            var_feats = encoders['var_type'].fit_transform(var_types)
        else:
            var_feats = encoders['var_type'].transform(var_types)
        features.append(var_feats)

    # 2. Allele Frequency (Biological - Continuous)
    if 'AlleleFrequency' in df.columns:
        af = pd.to_numeric(df['AlleleFrequency'], errors='coerce').values.reshape(-1, 1)
        if fit:
            # Impute missing AF with 0.0 (assuming missing = extremely rare/not seen)
            encoders['af_imputer'] = SimpleImputer(strategy='constant', fill_value=0.0)
            af_imputed = encoders['af_imputer'].fit_transform(af)
            
            encoders['af_scaler'] = StandardScaler()
            af_scaled = encoders['af_scaler'].fit_transform(af_imputed)
        else:
            af_imputed = encoders['af_imputer'].transform(af)
            af_scaled = encoders['af_scaler'].transform(af_imputed)
        features.append(af_scaled)

    # 3. CADD Score (Conservation/Pathogenicity - Continuous)
    if 'CADD_phred' in df.columns:
        cadd = pd.to_numeric(df['CADD_phred'], errors='coerce').values.reshape(-1, 1)
        if fit:
            # Impute missing CADD with the median of the training set
            encoders['cadd_imputer'] = SimpleImputer(strategy='median')
            cadd_imputed = encoders['cadd_imputer'].fit_transform(cadd)
            
            encoders['cadd_scaler'] = StandardScaler()
            cadd_scaled = encoders['cadd_scaler'].fit_transform(cadd_imputed)
        else:
            cadd_imputed = encoders['cadd_imputer'].transform(cadd)
            cadd_scaled = encoders['cadd_scaler'].transform(cadd_imputed)
        features.append(cadd_scaled)
    
    if len(features) > 0:
        X = np.hstack(features)
    else:
        # Fallback dummy
        X = np.zeros((len(df), 1))
        
    return X, encoders
