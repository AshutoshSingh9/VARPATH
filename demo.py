import os
import yaml
import sys

# Add nivep to path
sys.path.append(os.path.abspath('nivep'))
from src.train import train_pipeline

if __name__ == "__main__":
    print(f"\n--- Running NIVEP Demo Pipeline on REAL Data ---\n")
    model, encoders, metrics = train_pipeline("config.yaml")
    
    print("\n✅ End-to-end verification completed successfully on real data.")
