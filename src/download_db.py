import requests
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_variants(significance, size=1000):
    url = f"https://myvariant.info/v1/query"
    params = {
        "q": f"clinvar.rcv.clinical_significance:{significance}",
        "fields": "clinvar.rcv.clinical_significance,gnomad_exome.af.af,cadd.phred,snpeff.ann.effect",
        "size": size
    }
    logger.info(f"Fetching {size} {significance} variants from MyVariant.info...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    rows = []
    for hit in data.get('hits', []):
        # Extract significance
        sig = hit.get('clinvar', {}).get('rcv', {})
        if isinstance(sig, list):
            sig = sig[0].get('clinical_significance', 'Unknown')
        else:
            sig = sig.get('clinical_significance', 'Unknown')
            
        # Extract AF
        af = hit.get('gnomad_exome', {}).get('af', {}).get('af', None)
        
        # Extract CADD
        cadd = hit.get('cadd', {}).get('phred', None)
        
        # Extract Variant Type
        ann = hit.get('snpeff', {}).get('ann', [])
        if isinstance(ann, list) and len(ann) > 0:
            var_type = ann[0].get('effect', 'Unknown')
        elif isinstance(ann, dict):
            var_type = ann.get('effect', 'Unknown')
        else:
            var_type = 'Unknown'
            
        rows.append({
            'ClinicalSignificance': sig,
            'AlleleFrequency': af,
            'CADD_phred': cadd,
            'VariantType': var_type
        })
        
    return pd.DataFrame(rows)

def build_test_database(out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_pathogenic = fetch_variants("pathogenic", 1000)
    df_benign = fetch_variants("benign", 1000)
    
    df_combined = pd.concat([df_pathogenic, df_benign], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_combined.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_combined)} variants to {out_path}")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clinvar_raw.csv")
    build_test_database(out_path)
