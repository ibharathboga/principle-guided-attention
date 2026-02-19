
import os
import json
import matplotlib.pyplot as plt
import glob

def analyze():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    pattern = os.path.join(results_dir, "metrics_scale_*.json")
    files = glob.glob(pattern)
    
    # Structure: data[model][steps] = metrics
    data = {}
    
    for fpath in files:
        with open(fpath, 'r') as f:
            metrics = json.load(f)
            fname = os.path.basename(fpath)
            # metrics_scale_baseline_2500.json
            parts = fname.replace("metrics_scale_", "").replace(".json", "").split("_")
            
            if len(parts) == 2:
                model = parts[0]
                steps = int(parts[1])
            else:
                print(f"Skipping unknown file format: {fname}")
                continue
            
            if model not in data:
                data[model] = {}
            data[model][steps] = metrics

    # Generate Verdict Report
    report_lines = []
    report_lines.append("# Scaled PGA vs Baseline (d=64, L=64)")
    report_lines.append("")
    report_lines.append(f"Analysis of scaled experiment runs on Shakespeare.")
    report_lines.append("")
    
    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        
        colors = {'pga': 'blue', 'baseline': 'red'}
        
        for model, steps_dict in data.items():
            # Assume 2500 is the main run
            if 2500 in steps_dict:
                run = steps_dict[2500]
                color = colors.get(model, 'green')
                plt.plot(run['steps'], run['train_loss'], label=f'{model} (Train)', linestyle='-', color=color, alpha=0.7)
                plt.plot(run['steps'], run['val_loss'], label=f'{model} (Val)', linestyle='--', color=color, alpha=0.7)

        plt.title(f"Scaled Loss Comparison")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plot_path = f"scaled_comparison_plot.png"
        plt.savefig(os.path.join(results_dir, plot_path))
        plt.close()
        report_lines.append(f"![Scaled Comparison]({plot_path})")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"> [!WARNING] Could not generate plot: {e}")
    
    report_lines.append("| Model | Final Train Loss | Final Val Loss | Overfitting? |")
    report_lines.append("|-------|------------------|----------------|--------------|")
    
    for model, steps_dict in data.items():
        if 2500 in steps_dict:
            m = steps_dict[2500]
            final_train = m['train_loss'][-1]
            final_val = m['val_loss'][-1]
            gap = final_val - final_train
            overfitting = "Yes" if gap > 0.1 else "No"
            
            report_lines.append(f"| {model} | {final_train:.4f} | {final_val:.4f} | {overfitting} |")
    
    report_lines.append("")

    with open(os.path.join(results_dir, 'verdict_scaled.md'), 'w') as f:
        f.write('\n'.join(report_lines))
        
    print("Analysis complete. Scaled verdict report generated.")

if __name__ == "__main__":
    analyze()
