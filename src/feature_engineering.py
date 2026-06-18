import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def extract_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    Extracts tabular and biological features for the ML model.
    Imputers/scalers/encoders are fit on TRAIN only and reused on val/test,
    so there is no normalization leakage.

    Missingness in AlleleFrequency/CADD is strongly class-correlated in this
    ClinVar sample (pathogenic variants are far more often un-scored). We encode
    it as explicit AF_missing / CADD_missing indicator columns instead of letting
    it hide inside a magic imputation constant, and impute with the in-distribution
    median rather than an extreme sentinel. This makes the (real but ascertainment-
    driven) missingness signal auditable rather than disguised as a feature value.
    """
    if encoders is None:
        encoders = {}
    features = []

    # 1. Variant Type (categorical)
    if 'VariantType' in df.columns:
        var_types = df[['VariantType']].fillna('Unknown')
        if fit:
            encoders['var_type'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            var_feats = encoders['var_type'].fit_transform(var_types)
        else:
            var_feats = encoders['var_type'].transform(var_types)
        features.append(var_feats)

    # 2. Allele Frequency (continuous) + explicit missingness indicator
    if 'AlleleFrequency' in df.columns:
        af = pd.to_numeric(df['AlleleFrequency'], errors='coerce').values.reshape(-1, 1)
        af_missing = np.isnan(af).astype(float)
        if fit:
            encoders['af_imputer'] = SimpleImputer(strategy='median')
            af_imputed = encoders['af_imputer'].fit_transform(af)
            encoders['af_scaler'] = StandardScaler()
            af_scaled = encoders['af_scaler'].fit_transform(af_imputed)
        else:
            af_imputed = encoders['af_imputer'].transform(af)
            af_scaled = encoders['af_scaler'].transform(af_imputed)
        features.append(af_scaled)
        features.append(af_missing)

    # 3. CADD Score (continuous) + explicit missingness indicator
    if 'CADD_phred' in df.columns:
        cadd = pd.to_numeric(df['CADD_phred'], errors='coerce').values.reshape(-1, 1)
        cadd_missing = np.isnan(cadd).astype(float)
        if fit:
            encoders['cadd_imputer'] = SimpleImputer(strategy='median')
            cadd_imputed = encoders['cadd_imputer'].fit_transform(cadd)
            encoders['cadd_scaler'] = StandardScaler()
            cadd_scaled = encoders['cadd_scaler'].fit_transform(cadd_imputed)
        else:
            cadd_imputed = encoders['cadd_imputer'].transform(cadd)
            cadd_scaled = encoders['cadd_scaler'].transform(cadd_imputed)
        features.append(cadd_scaled)
        features.append(cadd_missing)

    if len(features) > 0:
        X = np.hstack(features)
    else:
        X = np.zeros((len(df), 1))

    return X, encoders
