"""Shared constants and helpers."""
import numpy as np

STAGES = ["fastqc", "trimming", "alignment", "sort_index", "markdup", "quant"]

STAGE_PARAMS = {
    "fastqc":     {"mu": 5.0, "sigma": 0.5},
    "trimming":   {"mu": 5.7, "sigma": 0.6},
    "alignment":  {"mu": 7.5, "sigma": 0.7},
    "sort_index": {"mu": 6.2, "sigma": 0.5},
    "markdup":    {"mu": 6.5, "sigma": 0.6},
    "quant":      {"mu": 5.9, "sigma": 0.5},
}

MERGE_PARAMS = {
    "multiqc": {"mu": 4.0, "sigma": 0.3},
    "deseq2":  {"mu": 5.5, "sigma": 0.4},
}

N_FEATURES = len(STAGES) + 1  # 6 one-hot + input_size_factor


def sample_runtime(stage, input_size_factor=1.0):
    p = STAGE_PARAMS.get(stage, MERGE_PARAMS.get(stage))
    if not p:
        raise ValueError(f"Unknown stage: {stage}")
    return max(1.0, np.random.lognormal(p["mu"], p["sigma"]) * np.sqrt(input_size_factor))


def encode_features(stage, input_size_factor):
    x = np.zeros(N_FEATURES)
    if stage in STAGES:
        x[STAGES.index(stage)] = 1.0
    x[6] = input_size_factor
    return x
