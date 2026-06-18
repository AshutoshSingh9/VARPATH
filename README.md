# VARPATH

**Variant Pathogenicity Assessment Research Tool**

A research-grade genomic variant classification system that predicts the pathogenicity of genetic variants. Built on an MLP with Monte Carlo Dropout for heuristic predictive-uncertainty estimates, integrated with live public genomic databases (gnomAD, ClinVar, CADD) via REST API. Note: performance is dominated by the CADD input feature (CADD-alone AUC ≈ 0.977), so the network adds little over a CADD threshold; treat the ACMG annotations as input-derived context, not as model reasoning.

---

## Features

- **Multi-source variant lookup** — Search by gene name, dbSNP rsID, or ClinVar RCV accession; variant parameters are auto-populated from live API calls to MyVariant.info
- **Pathogenicity prediction** — Binary classification (Pathogenic / Benign) with a probability score (uncalibrated; no reliability/Brier evaluation is performed)
- **Uncertainty estimation** — Monte Carlo Dropout over 20 stochastic forward passes yields a heuristic predictive-uncertainty estimate (σ) alongside each prediction
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
# 1. Navigate into the cloned repo
cd VARPATH

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

Evaluated on a **group-aware 3-way split** (≈60/20/20 train/val/test) of a
2,000-variant ClinVar-derived dataset. Identical feature vectors never straddle
the split, the checkpoint is selected on the validation set only, and the test
set is scored exactly once with a deterministic `eval()` pass (dropout off).
Reported as mean ± std over 10 split seeds.

| Metric | Value |
|---|---|
| **ROC-AUC (held-out test)** | **0.990 ± 0.012** |
| Accuracy | ~0.94 |
| F1 Score | ~0.91 |

> An earlier version of this README reported **AUC 0.996**. That figure was the
> single favorable `random_state=42` run with the test set doubling as the
> validation set and MC-dropout left on during scoring. It is not a stable
> estimate — the same (uncorrected) pipeline averages 0.991 ± 0.003 across seeds.

### Where the signal comes from

| Probe | AUC | Interpretation |
|---|---|---|
| Raw CADD score alone (observed rows) | 0.977 | Genuine variant biology; the model barely beats a CADD threshold |
| Full corrected model | 0.990 | In-distribution held-out performance |
| Missingness scrambled to be label-independent | 0.960 | **~0.03 of the AUC rides on class-correlated missingness** (a ClinVar ascertainment artifact, not biology) |
| Held out whole variant-consequence classes | 0.88 ± 0.14 | Worst-case generalization to unseen variant types is far weaker and unstable |

**Honest takeaway:** the ~0.99 is real but dominated by CADD plus an ascertainment
artifact. The dataset is a balanced case/control ClinVar sample, **not** an
ascertainment-matched prospective stream, so this number will not transfer
unchanged to clinical deployment. There is no gene/locus identifier in the data,
so a true gene-disjoint (leakage-free) generalization estimate cannot be computed.

---

## Project Structure

```
VARPATH/
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
