# VARPATH

**Variant Pathogenicity Assessment Research Tool**

A clinical-grade genomic variant classification system that predicts the pathogenicity of genetic variants using a Bayesian-approximate neural network. Built on an MLP with Monte Carlo Dropout for uncertainty quantification, integrated with live public genomic databases (gnomAD, ClinVar, CADD) via REST API.

---

## Features

- **Multi-source variant lookup** — Search by gene name, dbSNP rsID, or ClinVar RCV accession; variant parameters are auto-populated from live API calls to MyVariant.info
- **Pathogenicity prediction** — Binary classification (Pathogenic / Benign) with calibrated probability score
- **Uncertainty estimation** — Monte Carlo Dropout over 20 stochastic forward passes yields an epistemic uncertainty bound (σ) alongside each prediction
- **Best-checkpoint training** — Trains for 100 epochs, tracking validation loss per epoch and automatically restoring the best-generalizing weights before saving
- **ACMG-aligned interpretation** — Results are annotated with ACMG/AMP guideline context (BA1, PM2 criteria) for allele frequency and CADD thresholds
- **Interactive UI** — Dark-themed, single-page Streamlit dashboard with an animated dot-field background

---

## Stack

| Layer | Technology |
|---|---|
| Model | PyTorch MLP, Monte Carlo Dropout |
| Feature engineering | scikit-learn (OHE, StandardScaler, SimpleImputer) |
| Data | ClinVar synthetic subset (2,000 variants) |
| API integration | MyVariant.info, gnomAD, CADD |
| UI | Streamlit |
| Package management | uv |

---

## Quickstart

```bash
# 1. Navigate into the project
cd nivep

# 2. Install dependencies (uv manages the virtual environment automatically)
uv pip install -r requirements.txt

# 3. Train the model — runs 100 epochs, saves best checkpoint to models/
uv run python -m src.train

# 4. Launch the dashboard
uv run streamlit run app/app.py

# 5. Run tests
uv run python -m pytest
```

---

## Model Performance

Trained on a 80/20 split of a 2,000-variant ClinVar-derived dataset.
Best checkpoint restored by minimum validation loss.

| Metric | Value |
|---|---|
| Accuracy | 0.975 |
| F1 Score | 0.975 |
| AUC-ROC | 0.996 |
| Val Loss (best) | 0.087 |

---

## Project Structure

```
nivep/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── data_processing.py  # ClinVar loading + cleaning
│   ├── feature_engineering.py  # OHE, scaling, imputation
│   ├── model.py            # MLP with MC Dropout
│   ├── train.py            # Training pipeline + checkpointing
│   ├── evaluate.py         # Accuracy, F1, AUC metrics
│   └── uncertainty.py      # Monte Carlo Dropout inference
├── tests/
│   └── test_pipeline.py    # Feature engineering unit tests
├── models/                 # Saved weights + encoders (git-ignored)
├── data/                   # Raw ClinVar CSV (git-ignored)
├── config.yaml             # Hyperparameters
└── requirements.txt
```

---

## API Lookup Examples

| Method | Input | Expected Result |
|---|---|---|
| rsID | `rs121913527` | Pathogenic (BRAF V600E) |
| rsID | `rs58991260` | Benign |
| ClinVar | `RCV000013961` | Pathogenic |
| Gene search | `TP53` | Multiple variants dropdown |
| Manual | `stop_gained, AF=0.00, CADD=45.0` | Pathogenic |
| Manual | `synonymous_variant, AF=0.10, CADD=2.5` | Benign |

---

## Allele Frequency Reference (ACMG/AMP)

| AF Range | Classification | ACMG Criterion |
|---|---|---|
| > 5% | Benign standalone | BA1 |
| 1 – 5% | Benign supporting | BS1 |
| 0.1 – 1% | Uncertain significance | — |
| 0.01 – 0.1% | Pathogenic moderate | PM2 |
| < 0.01% or absent | Pathogenic prerequisite | PM2 |

---

## License

MIT
