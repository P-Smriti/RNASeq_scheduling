"""Collect 450 training tasks from Nextflow traces, WfCommons, and synthetic generation."""
import os, json
import numpy as np
import pandas as pd
from utils import STAGES, sample_runtime

STAGE_MAP = {
    "NFCORE_RNASEQ:FASTQC": "fastqc",
    "NFCORE_RNASEQ:TRIMGALORE": "trimming",
    "NFCORE_RNASEQ:STAR_ALIGN": "alignment",
    "NFCORE_RNASEQ:SAMTOOLS_SORT": "sort_index",
    "NFCORE_RNASEQ:PICARD_MARKDUP": "markdup",
    "NFCORE_RNASEQ:SALMON_QUANT": "quant",
}

SAMPLE_READS = {
    "SRR1234501": 35, "SRR1234502": 28, "SRR1234503": 42,
    "SRR1234504": 20, "SRR1234505": 15, "SRR1234506": 50,
    "SRR1234507": 38, "SRR1234508": 12, "SRR1234509": 18,
    "SRR1234510": 45,
}


def extract_real_traces(trace_dir="data/nextflow_traces"):
    """Parse Nextflow trace files into training rows."""
    rows = []
    trace_files = sorted([f for f in os.listdir(trace_dir) if f.startswith("trace_") and f.endswith(".txt")])

    for tf in trace_files:
        sid = tf.replace("trace_", "").replace(".txt", "")
        sf = SAMPLE_READS.get(sid, 20) / 20.0

        with open(os.path.join(trace_dir, tf)) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < len(header):
                    continue
                row = dict(zip(header, fields))
                proc = row.get("name", "").split("(")[0].strip()
                stage = STAGE_MAP.get(proc)
                if not stage:
                    continue
                rt = float(row.get("duration", "0s").replace("s", ""))
                rows.append({"stage": stage, "input_size_factor": round(sf, 3), "runtime": round(rt, 1), "source": "real"})

    print(f"  Real: {len(rows)} tasks from {len(trace_files)} traces")
    return pd.DataFrame(rows)


def extract_wfcommons_traces(wfc_dir="data/wfcommons_traces", max_tasks=90):
    """Parse WfCommons JSON files into training rows."""
    rows = []
    jfiles = sorted([f for f in os.listdir(wfc_dir) if f.endswith(".json")])

    for jf in jfiles:
        with open(os.path.join(wfc_dir, jf)) as f:
            wf = json.load(f)
        tasks = wf.get("workflow", {}).get("execution", {}).get("tasks", [])
        # parse read count from description
        reads_m = 20
        for tok in wf.get("description", "").replace(",", "").split():
            if tok.endswith("M"):
                try: reads_m = int(tok.replace("M", ""))
                except: pass
        sf = reads_m / 20.0
        for t in tasks:
            cat = t.get("category", t.get("name", ""))
            stage = next((s for s in STAGES if s in cat), None)
            if not stage:
                continue
            rt = t.get("runtime", 0) * 50  # scale test-profile to production
            rows.append({"stage": stage, "input_size_factor": round(sf, 3), "runtime": round(rt, 1), "source": "wfcommons"})

    df = pd.DataFrame(rows).head(max_tasks)
    print(f"  WfCommons: {len(df)} tasks from {len(jfiles)} files")
    return df


def generate_synthetic(n_wf=50, seed=200):
    """Generate synthetic tasks with lognormal distributions."""
    np.random.seed(seed)
    rows = []
    for _ in range(n_wf):
        rc = np.random.choice([5, 10, 20, 50, 100])
        sf = (rc / 20.0) * np.random.uniform(0.7, 1.3)
        for stage in STAGES:
            rt = sample_runtime(stage, sf)
            rows.append({"stage": stage, "input_size_factor": round(sf, 3), "runtime": round(rt, 1), "source": "synthetic"})
    print(f"  Synthetic: {len(rows)} tasks from {n_wf} workflows")
    return pd.DataFrame(rows)


def generate_all_data(output_path="data/training_data.csv"):
    """Combine all sources and save."""
    print("Collecting training data:")
    real = extract_real_traces()
    wfc = extract_wfcommons_traces()
    syn = generate_synthetic()
    df = pd.concat([real, wfc, syn], ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Total: {len(df)} tasks -> {output_path}")
    return df

if __name__ == "__main__":
    generate_all_data()
