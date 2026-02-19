
import os
import json
import matplotlib.pyplot as plt
import glob

def analyze():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    pattern = os.path.join(results_dir, "metrics_*.json")
    files = glob.glob(pattern)
    
    # Structure: data[dataset][model][steps] = metrics
    data = {}
    
    for fpath in files:
        with open(fpath, 'r') as f:
            metrics = json.load(f)
            fname = os.path.basename(fpath)
            # metrics_pga_names_500.json or metrics_baseline_names_500.json
            # parts = fname.replace("metrics_", "").replace(".json", "").split("_")
            # If simple split:
            # metrics_pga_names_500 -> ['pga', 'names', '500']
            # metrics_names_500 -> ['names', '500'] (Legacy from before model arg)
            
            clean_name = fname.replace("metrics_", "").replace(".json", "")
            parts = clean_name.split("_")
            
            if len(parts) == 2:
                # Legacy: implied 'pga'
                model = 'pga'
                dataset = parts[0]
                steps = int(parts[1])
            elif len(parts) == 3:
                model = parts[0]
                dataset = parts[1]
                steps = int(parts[2])
            else:
                print(f"Skipping unknown file format: {fname}")
                continue
                
            if dataset not in data:
                data[dataset] = {}
            if model not in data[dataset]:
                data[dataset][model] = {}
            
            data[dataset][model][steps] = metrics

    # Generate Verdict Report
    report_lines = []
    report_lines.append("# Comparison: PGA vs Baseline")
    report_lines.append("")
    report_lines.append(f"Analysis of {len(files)} experiment runs.")
    report_lines.append("")
    
    for dataset, models_dict in data.items():
        report_lines.append(f"## Dataset: {dataset.title()}")
        
        # Plotting
        try:
            plt.figure(figsize=(10, 6))
            
            colors = {'pga': 'blue', 'baseline': 'red'}
            
            for model, steps_dict in models_dict.items():
                sorted_steps = sorted(steps_dict.keys())
                
                # We want to plot the longest run for each model to show the full curve?
                # Or plot final losses? 
                # Let's plot the longest run's Loss Curve.
                max_step = sorted_steps[-1]
                long_run = steps_dict[max_step]
                
                color = colors.get(model, 'green')
                plt.plot(long_run['steps'], long_run['train_loss'], label=f'{model} (Train)', linestyle='-', color=color, alpha=0.6)
                plt.plot(long_run['steps'], long_run['val_loss'], label=f'{model} (Val)', linestyle='--', color=color, alpha=0.6)

            plt.title(f"Loss Comparison - {dataset} (Max Steps)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plot_path = f"comparison_plot_{dataset}.png"
            plt.savefig(os.path.join(results_dir, plot_path))
            plt.close()
            report_lines.append(f"![Comparison Plot]({plot_path})")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"> [!WARNING] Could not generate plot: {e}")
        
        report_lines.append("| Model | Steps | Final Train Loss | Final Val Loss | Overfitting? |")
        report_lines.append("|-------|-------|------------------|----------------|--------------|")
        
        for model, steps_dict in models_dict.items():
            sorted_steps = sorted(steps_dict.keys())
            for steps in sorted_steps:
                m = steps_dict[steps]
                final_train = m['train_loss'][-1]
                final_val = m['val_loss'][-1]
                gap = final_val - final_train
                overfitting = "Yes" if gap > 0.1 else "No"
                
                report_lines.append(f"| {model} | {steps} | {final_train:.4f} | {final_val:.4f} | {overfitting} |")
        
        report_lines.append("")

    with open(os.path.join(results_dir, 'verdict_report_comparison.md'), 'w') as f:
        f.write('\n'.join(report_lines))
        
    print("Analysis complete. Comparison report generated.")

if __name__ == "__main__":
    analyze()
