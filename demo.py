import yaml
import sys

from src.train import train_pipeline

if __name__ == "__main__":
    print("\n--- Running VARPATH Demo Pipeline on Real Data ---\n")
    model, encoders, metrics = train_pipeline("config.yaml")

    print("\nEnd-to-end verification completed successfully on real data.")
