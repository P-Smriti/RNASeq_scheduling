# Runtime Prediction-Aware Scheduling for RNA-seq Workflows

EECS 700 (Algorithms for HPC) project by Smriti Pranjal, University of Kansas.

Predicts RNA-seq task runtimes using ML (Linear, Ridge, Lasso, SVR, Random Forest) and uses the predictions in a priority list scheduler with round-up reservation to reduce makespan on homogeneous HPC processors.

## How to run

```
pip install -r requirements.txt
python src/run_experiments.py
```

## Project structure

- `src/` — data extraction, prediction models, scheduler, experiment runner
- `data/` — Nextflow traces (10 real cases), WfCommons JSONs (15 workflows), synthetic data
- `results/` — generated figures and summary
- `docs/report.md` — full project report
