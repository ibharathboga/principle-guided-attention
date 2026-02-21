import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help="Baseline log file (txt/csv)")
    parser.add_argument('--pga', type=str, required=True, help="PGA JIT log file (csv)")
    args = parser.parse_args()

    # Load Baseline
    try:
        df_base = pd.read_csv(args.baseline, names=['step', 'loss'], header=0)
    except:
        print("Could not load baseline log")
        return

    # Load PGA
    try:
        df_pga = pd.read_csv(args.pga)
    except:
        print("Could not load PGA log")
        return

    plt.figure(figsize=(12, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(df_base['step'], df_base['loss'], label='Baseline Train', alpha=0.7)
    plt.plot(df_pga['step'], df_pga['train_loss'], label='PGA Train', alpha=0.7)
    plt.plot(df_pga['step'], df_pga['val_loss'], label='PGA Val', linestyle='--')
    plt.title("Loss Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Essence Norm Plot
    plt.subplot(1, 2, 2)
    plt.plot(df_pga['step'], df_pga['essence_norm'], color='purple')
    plt.title("Essence Vector Norm (L2)")
    plt.xlabel("Steps")
    plt.ylabel("Norm")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pga_jit_results.png")
    print("Saved plot to pga_jit_results.png")

if __name__ == "__main__":
    main()
