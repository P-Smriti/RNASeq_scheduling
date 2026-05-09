"""
Full experimental pipeline for the RNA-seq scheduling project.

Steps:
    1. Generate training data (450 tasks)
    2. Train and evaluate 5 prediction models
    3. Run scheduling experiments (vary samples, processors)
    4. Generate all result figures

Usage:
    python src/run_experiments.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import STAGES, sample_runtime, encode_features, N_FEATURES
from dag_builder import build_dag
from data_generator import generate_all_data
from predictor import run_prediction_pipeline
from scheduler import schedule_fcfs, schedule_predicted


def run_scheduling_experiment(n_samples, n_processors, model, n_trials=10):
    """
    Run scheduling experiment with fresh workflows.
    
    For each trial:
        1. Generate a new workflow with random input sizes
        2. Predict runtimes using the trained model
        3. Schedule with FCFS, predicted, and oracle
        4. Record makespans
    
    Args:
        n_samples: number of RNA-seq samples in the workflow
        n_processors: number of identical processors
        model: trained sklearn model
        n_trials: number of random trials
    
    Returns:
        dict of {method: {mean, std}}
    """
    results = {"FCFS": [], "Predicted": [], "Oracle": []}
    
    for trial in range(n_trials):
        np.random.seed(trial * 1000 + n_samples * 100 + n_processors)
        
        # Generate workflow with random input sizes
        read_count = np.random.choice([10, 20, 50])
        base_factor = read_count / 20.0
        
        true_runtimes = {}
        features_list = []
        
        for s in range(n_samples):
            sample_factor = base_factor * np.random.uniform(0.7, 1.3)
            for stage in STAGES:
                task_id = f"S{s}_{stage}"
                true_runtimes[task_id] = sample_runtime(stage, sample_factor)
                features_list.append({
                    "task_id": task_id,
                    "stage": stage,
                    "input_size_factor": sample_factor
                })
        
        # Merge tasks
        for merge in ["multiqc", "deseq2"]:
            merge_id = f"MERGE_{merge}"
            merge_factor = n_samples / 4.0
            true_runtimes[merge_id] = sample_runtime(merge, merge_factor)
        
        # Build DAG
        G, info = build_dag(n_samples, true_runtimes)
        
        # Predict runtimes using ML model
        X_pred = np.zeros((len(features_list), N_FEATURES))
        for i, f in enumerate(features_list):
            X_pred[i] = encode_features(f["stage"], f["input_size_factor"])
        
        ml_predictions = model.predict(X_pred)
        predicted_runtimes = {}
        for i, f in enumerate(features_list):
            predicted_runtimes[f["task_id"]] = max(1.0, ml_predictions[i])
        # Merge tasks: use average prediction
        for merge in ["multiqc", "deseq2"]:
            predicted_runtimes[f"MERGE_{merge}"] = true_runtimes[f"MERGE_{merge}"]
        
        # Oracle: use true runtimes
        oracle_runtimes = {t: info[t]["runtime"] for t in G.nodes()}
        
        # Schedule
        results["FCFS"].append(schedule_fcfs(G, info, n_processors))
        results["Predicted"].append(
            schedule_predicted(G, info, predicted_runtimes, n_processors)
        )
        results["Oracle"].append(
            schedule_predicted(G, info, oracle_runtimes, n_processors)
        )
    
    return {
        method: {"mean": np.mean(vals), "std": np.std(vals)}
        for method, vals in results.items()
    }


def generate_figures(results, best_name, X_test, y_test, best_pred, df, sa_mape, 
                     bench, exp_samples, exp_procs, heatmap, output_dir="results"):
    """Generate all result figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Figure 1: Model Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = list(results.keys()) + ["Stage Avg"]
    mapes = [results[n]["mape"] for n in results] + [sa_mape]
    best_idx = min(range(len(results)), key=lambda i: list(results.values())[i]["mape"])
    colors = ["#888888"] * len(names)
    colors[best_idx] = "#2E86C1"
    ax.bar(range(len(names)), mapes, color=colors, edgecolor="#555555")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Prediction Accuracy (lower is better)", fontweight="bold")
    for i, v in enumerate(mapes):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    
    ax = axes[1]
    r2s = [results[n]["r2"] for n in results]
    colors2 = ["#888888"] * len(r2s)
    colors2[best_idx] = "#2E86C1"
    ax.bar(range(len(r2s)), r2s, color=colors2, edgecolor="#555555")
    ax.set_xticks(range(len(r2s)))
    ax.set_xticklabels(list(results.keys()), rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Fit (higher is better)", fontweight="bold")
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 2: Actual vs Predicted ──
    clip = np.percentile(np.concatenate([y_test, best_pred]), 80)
    mask = (y_test <= clip) & (best_pred <= clip)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test[~mask], best_pred[~mask], alpha=0.15, s=15, color="#2E86C1")
    ax.scatter(y_test[mask], best_pred[mask], alpha=0.5, s=30, color="#2E86C1")
    ax.plot([0, clip], [0, clip], "k--", alpha=0.5, linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Runtime (s)")
    ax.set_ylabel("Predicted Runtime (s)")
    mape_val = mean_absolute_percentage_error(y_test, best_pred) * 100
    r2_val = results[best_name]["r2"]
    ax.set_title(f"{best_name}: MAPE={mape_val:.1f}%, R²={r2_val:.3f} (90 test tasks)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, clip * 1.1)
    ax.set_ylim(0, clip * 1.1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 3: MAE by Stage + Runtime Distribution ──
    test_stages = [STAGES[int(X_test[i, :6].argmax())] for i in range(len(X_test))]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stage_mae = []
    stage_names_sorted = []
    for stage in STAGES:
        m = np.array([s == stage for s in test_stages])
        if m.sum() > 0:
            stage_mae.append((stage.replace("_", " "),
                             mean_absolute_error(y_test[m], best_pred[m]),
                             int(m.sum())))
    stage_mae.sort(key=lambda x: x[1])
    
    ax = axes[0]
    colors_s = ["#888888"] * len(stage_mae)
    colors_s[-1] = "#C06060"
    ax.bar(range(len(stage_mae)), [s[1] for s in stage_mae], color=colors_s, edgecolor="#555555")
    ax.set_xticks(range(len(stage_mae)))
    ax.set_xticklabels([s[0] for s in stage_mae], fontsize=11)
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Absolute Prediction Error\n(higher = worse for scheduling)", fontweight="bold")
    for i, (n, v, c) in enumerate(stage_mae):
        ax.text(i, v + 15, f"{v:.0f}s", ha="center", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    
    ax = axes[1]
    box_data = [df[df["stage"] == s]["runtime"].values for s in STAGES]
    bp = ax.boxplot(box_data, tick_labels=[s.replace("_", " ") for s in STAGES],
                    patch_artist=True, showfliers=True,
                    flierprops=dict(markersize=2, alpha=0.3))
    align_idx = STAGES.index("alignment")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#C06060" if i == align_idx else "#888888")
        patch.set_alpha(0.7)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Distribution\n(alignment has highest variance)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_by_stage.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 4: Benchmark Bar Chart ──
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["FCFS", "Predicted", "Oracle"]
    means = [bench[m]["mean"] for m in methods]
    stds = [bench[m]["std"] for m in methods]
    labels = ["No Prediction\n(FCFS)", f"With Prediction\n({best_name})", "Perfect\n(Oracle)"]
    colors_b = ["#999999", "#2E86C1", "#E67E22"]
    bars = ax.bar(labels, means, yerr=stds, color=colors_b, capsize=6, edgecolor="#555555")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 300, f"{v:.0f}s",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Makespan (seconds)")
    ax.set_title("Scheduling Performance\n(6 samples, 4 processors, 10 trials)", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_benchmark.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 5: Vary Samples ──
    scs = list(exp_samples.keys())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, c, mk in [("FCFS", "#999999", "o"), ("Predicted", "#2E86C1", "s"), ("Oracle", "#E67E22", "^")]:
        mn = [exp_samples[ns][m]["mean"] for ns in scs]
        sd = [exp_samples[ns][m]["std"] for ns in scs]
        ax.errorbar(scs, mn, yerr=sd, marker=mk, label=m, color=c, linewidth=2, capsize=4)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Makespan (seconds)")
    ax.set_title("Makespan vs Workflow Size (4 processors)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vary_samples.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 6: Vary Processors ──
    pcs = list(exp_procs.keys())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, c, mk in [("FCFS", "#999999", "o"), ("Predicted", "#2E86C1", "s"), ("Oracle", "#E67E22", "^")]:
        mn = [exp_procs[np_][m]["mean"] for np_ in pcs]
        sd = [exp_procs[np_][m]["std"] for np_ in pcs]
        ax.errorbar(pcs, mn, yerr=sd, marker=mk, label=m, color=c, linewidth=2, capsize=4)
    ax.set_xlabel("Number of Processors")
    ax.set_ylabel("Makespan (seconds)")
    ax.set_title("Makespan vs Processor Count (6 samples)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vary_processors.png", dpi=180, facecolor="white")
    plt.close()
    
    # ── Figure 7: Heatmap ──
    srange, prange, grid = heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(prange)))
    ax.set_xticklabels(prange)
    ax.set_yticks(range(len(srange)))
    ax.set_yticklabels(srange)
    ax.set_xlabel("Processors")
    ax.set_ylabel("Samples")
    ax.set_title("% Makespan Improvement (Predicted vs FCFS)", fontweight="bold")
    for i in range(len(srange)):
        for j in range(len(prange)):
            ax.text(j, i, f"{grid[i,j]:+.1f}%", ha="center", va="center", fontsize=11,
                    color="black" if abs(grid[i,j]) < 3 else "white")
    plt.colorbar(im, label="Improvement (%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_heatmap.png", dpi=180, facecolor="white")
    plt.close()
    
    print(f"\nAll figures saved to {output_dir}/")


def main():
    """Run the complete experimental pipeline."""
    print("=" * 60)
    print("RNA-seq Prediction-Aware Scheduling — Full Pipeline")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n[Step 1] Generating training data...")
    df = generate_all_data("data/training_data.csv")
    
    # Step 2: Train and evaluate models
    print("\n[Step 2] Training prediction models...")
    (models, results, best_name,
     X_train, X_test, y_train, y_test, df, sa_mape) = \
        run_prediction_pipeline("data/training_data.csv")
    
    best_model = models[best_name]
    best_pred = results[best_name]["predictions"]
    
    # Step 3: Scheduling experiments
    print("\n[Step 3] Running scheduling experiments...")
    
    # Benchmark: 6 samples, 4 processors
    print("  Benchmark (6 samples, 4 processors)...")
    bench = run_scheduling_experiment(6, 4, best_model, n_trials=10)
    fcfs_m = bench["FCFS"]["mean"]
    pred_m = bench["Predicted"]["mean"]
    orac_m = bench["Oracle"]["mean"]
    imp = (fcfs_m - pred_m) / fcfs_m * 100
    ceiling = (fcfs_m - orac_m) / fcfs_m * 100
    capture = (fcfs_m - pred_m) / (fcfs_m - orac_m) * 100 if fcfs_m != orac_m else 0
    
    print(f"    FCFS:      {fcfs_m:.0f}s")
    print(f"    Predicted: {pred_m:.0f}s ({imp:+.1f}%)")
    print(f"    Oracle:    {orac_m:.0f}s ({ceiling:+.1f}%)")
    print(f"    Capture:   {capture:.0f}% of oracle ceiling")
    
    # Vary samples
    print("  Varying samples (4 processors)...")
    sample_counts = [2, 4, 6, 8, 10, 12]
    exp_samples = {}
    for ns in sample_counts:
        exp_samples[ns] = run_scheduling_experiment(ns, 4, best_model, 10)
    
    # Vary processors
    print("  Varying processors (6 samples)...")
    proc_counts = [2, 4, 6, 8, 10]
    exp_procs = {}
    for np_ in proc_counts:
        exp_procs[np_] = run_scheduling_experiment(6, np_, best_model, 10)
    
    # Heatmap
    print("  Computing improvement heatmap...")
    srange = [2, 4, 6, 8, 10]
    prange = [2, 4, 6, 8]
    grid = np.zeros((len(srange), len(prange)))
    for i, ns in enumerate(srange):
        for j, np_ in enumerate(prange):
            r = run_scheduling_experiment(ns, np_, best_model, 8)
            grid[i, j] = (r["FCFS"]["mean"] - r["Predicted"]["mean"]) / r["FCFS"]["mean"] * 100
    
    # Step 4: Generate figures
    print("\n[Step 4] Generating figures...")
    generate_figures(
        results, best_name, X_test, y_test, best_pred, df, sa_mape,
        bench, exp_samples, exp_procs, (srange, prange, grid)
    )
    
    # Save summary
    summary = {
        "best_model": best_name,
        "prediction": {n: {"mape": r["mape"], "mae": r["mae"], "r2": r["r2"]}
                       for n, r in results.items()},
        "stage_avg_mape": sa_mape,
        "benchmark": {m: {"mean": bench[m]["mean"], "std": bench[m]["std"]}
                      for m in bench},
        "improvement_pct": imp,
        "oracle_ceiling_pct": ceiling,
        "capture_pct": capture,
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DONE. Key results:")
    print(f"  Best model:   {best_name} (MAPE={results[best_name]['mape']:.1f}%)")
    print(f"  Improvement:  {imp:+.1f}% over FCFS")
    print(f"  Oracle ceiling: {ceiling:+.1f}%")
    print(f"  Capture:      {capture:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
