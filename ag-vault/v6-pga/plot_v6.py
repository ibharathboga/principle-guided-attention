import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--pga', type=str, required=True)
    args = parser.parse_args()

    try:
        df_base = pd.read_csv(args.baseline)
        df_pga = pd.read_csv(args.pga)
    except Exception as e:
        print(f"Error reading logs: {e}")
        return

    plt.figure(figsize=(10, 6))
    
    # Smoothing for better visualization
    smooth_window = 20 # More smoothing for longer run/more noise?
    
    plt.plot(df_base['step'], df_base['loss'], label='Baseline Scale', alpha=0.3, color='cyan')
    plt.plot(df_base['step'], df_base['loss'].rolling(smooth_window).mean(), label='Baseline Scale (Smoothed)', color='blue')
    
    plt.plot(df_pga['step'], df_pga['loss'], label='PGA Scale', alpha=0.3, color='lime')
    plt.plot(df_pga['step'], df_pga['loss'].rolling(smooth_window).mean(), label='PGA Scale (Smoothed)', color='green')
    
    plt.title("v6-pga Scale Up Benchmark (2500 Steps)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    outfile = "v6_benchmark_scale.png"
    plt.savefig(outfile)
    print(f"Saved plot to {outfile}")

if __name__ == "__main__":
    main()
