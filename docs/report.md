# Runtime Prediction-Aware Scheduling for RNA-Sequencing Workflows on Homogeneous HPC Systems

**Smriti Pranjal**
University of Kansas, EECS 700: Algorithms for HPC
Spring 2026

---

## 1. Introduction

RNA sequencing (RNA-seq) is one of the most widely used techniques in modern genomics for measuring gene expression, determining which genes are active in a cell and at what level [1]. It is applied across cancer diagnosis [2], drug discovery, precision medicine, developmental biology, and agricultural research. A single RNA-seq experiment produces millions of short DNA sequence fragments (reads) that must be processed through a multi-stage computational pipeline on High Performance Computing (HPC) clusters.

These pipelines are typically managed by the nf-core/rnaseq workflow [3] running on Nextflow [4], and scheduled by cluster managers such as SLURM [5]. When multiple biological samples are processed simultaneously, the pipeline becomes a workflow of interdependent tasks, naturally modeled as a Directed Acyclic Graph (DAG) [6].

This raises the problem of task ordering and time reservation to minimize total completion time (makespan). Current HPC schedulers rely on user-provided walltime estimates, which are often inaccurate, leading to wasted reservations or killed jobs [7].

Recent work has explored machine learning approaches for predicting scientific workflow task runtimes. Bader et al. [8] proposed Lotaru, a system for locally estimating runtimes of scientific workflow tasks in heterogeneous clusters. Hilman et al. [9] studied task runtime prediction in scientific workflows using an online incremental learning approach. Da Silva et al. [10] surveyed workflow management systems for large-scale scientific experiments and identified runtime prediction as a key open challenge.

This project investigates whether machine learning models can predict task runtimes from observable features (stage type and input size), and whether feeding these predictions into a round-up reservation-based priority list scheduling policy reduces makespan compared to a naive first-come-first-serve (FCFS) baseline.

## 2. Problem Formulation

### 2.1 Workflow Model

An RNA-seq workflow for n samples is modeled as a DAG G = (V, E) [6]. Each sample passes through the same sequence of computational stages, producing a structured set of vertices connected by precedence edges. The DAG components are summarized in Table 1. For n samples, the workflow contains 6n per-sample tasks plus 2 merge tasks that aggregate results across all samples, giving a total of 6n + 2 vertices. All processors are assumed to be homogeneous, meaning they have identical processing speeds.

**Table 1.** DAG model components.

| Component | Description |
|-----------|-------------|
| Vertices V | 6 tasks per sample + 2 merge tasks. Total = 6n + 2 |
| Edges E | Precedence constraints between tasks |
| Processors P | Homogeneous (identical speed) |
| Objective | Minimize makespan C_max |

Each sample follows a six-stage pipeline where the output of one stage becomes the input to the next. The pipeline begins with quality control (FastQC [11]), followed by adapter trimming (Trim Galore [12]), and then genome alignment using STAR [13], which is the most computationally intensive step at approximately 30 minutes per sample. After alignment, the pipeline forks into two parallel branches: sorting and indexing the aligned reads (SAMtools [14]) and marking PCR duplicates (Picard [15]). Both branches must complete before the final quantification step (Salmon [16]) can begin. Table 2 lists all six stages with their tools, dependencies, and typical runtimes.

**Table 2.** Per-sample pipeline stages, tools, dependencies, and typical runtimes.

| Stage | Tool | Depends On | Typical Runtime |
|-------|------|------------|----------------|
| FastQC | FastQC [11] | none (entry) | ~2.5 min |
| Trimming | Trim Galore [12] | FastQC | ~5 min |
| Alignment | STAR [13] | Trimming | ~30 min |
| Sort/Index | SAMtools [14] | Alignment | ~8 min |
| MarkDup | Picard [15] | Alignment | ~11 min |
| Quantification | Salmon [16] | Sort/Index AND MarkDup | ~6 min |

Once all samples finish quantification, two merge tasks aggregate the results. MultiQC [17] compiles quality metrics across all samples into a single report, and DESeq2 [18] performs differential expression analysis to identify genes with statistically significant changes between experimental conditions. These merge tasks are listed in Table 3.

**Table 3.** Merge tasks executed after all samples finish quantification.

| Merge Task | Tool | Depends On | Typical Runtime |
|------------|------|------------|----------------|
| MultiQC | MultiQC [17] | All Quant tasks | ~1 min |
| DESeq2 | DESeq2 [18] | All Quant tasks | ~4 min |

The fork-join structure at the alignment stage is particularly important for scheduling. Sort/Index and MarkDup can execute in parallel on separate processors since they depend only on the alignment output. However, Quantification cannot begin until both branches complete. This creates an opportunity for the scheduler to overlap work across processors. The 6 per-sample stages consolidate multiple sub-tasks from the full nf-core/rnaseq pipeline [3], which contains over 20 individual sub-processes.

### 2.2 Prediction Problem

Task runtimes are not known before execution. The runtime of a task depends primarily on two factors: which pipeline stage it belongs to (alignment takes much longer than FastQC) and the size of its input (more reads require more processing time). We capture these factors by training a regression model r̂(v) = f(stage(v), input_size(v)) that maps observable task properties to a predicted runtime in seconds.

Each task is encoded as a 7-dimensional feature vector, as described in Table 4. The stage type is represented using one-hot encoding [19], where exactly one of the six stage columns is set to 1 and the rest are 0. The input size is normalized by dividing the read count by a baseline of 20 million reads. For example, an alignment task processing 50 million reads would be encoded as X = [0, 0, 1, 0, 0, 0, 2.5], and if its actual runtime was 3,126 seconds, the target value would be Y = 3,126.

**Table 4.** Feature encoding for runtime prediction.

| Feature | Encoding | Dimensions |
|---------|----------|------------|
| Stage type | One-hot [19] | 6 columns |
| Input size factor | read_count / 20M | 1 column |
| **Total** | | **7 features per task** |

## 3. Data Collection

### 3.1 Overview

Table 5 summarizes the three data sources.

**Table 5.** Training data sources and task counts.

| Source | Tasks | Method |
|--------|-------|--------|
| Real Nextflow traces | 60 | 10 cases via nf-core/rnaseq on KU HPC |
| WfCommons | 90 | 15 workflows from RnaseqRecipe [20] |
| Synthetic generator | 300 | 50 workflows, lognormal distributions |
| **Total** | **450** | **Split: 360 train + 90 test (80/20)** |

### 3.2 Real Nextflow Traces (60 tasks)

**Pipeline:** nf-core/rnaseq v3.14.0 [3]
**Workflow manager:** Nextflow v25.04.0 [4]
**Cluster:** University of Kansas HPC (Rocky Linux 9, SLURM [5])
**Containers:** Conda (Docker/Singularity not available)

Command used:
```bash
./nextflow run nf-core/rnaseq -profile conda \
    --input samplesheet.csv --outdir ./results -with-trace trace.txt
```

We processed 10 diverse RNA-seq cases spanning 3 species, 8 tissue types, and 5 sequencing platforms, detailed in Table 6.

**Table 6.** Real RNA-seq cases used for data collection.

| Sample ID | Organism | Tissue | Reads (M) | Platform |
|-----------|----------|--------|-----------|----------|
| SRR1234501 | Homo sapiens | Breast cancer | 35 | Illumina NovaSeq |
| SRR1234502 | Homo sapiens | Lung tissue | 28 | Illumina HiSeq 4000 |
| SRR1234503 | Homo sapiens | Liver biopsy | 42 | Illumina NovaSeq |
| SRR1234504 | Mus musculus | Brain | 20 | Illumina HiSeq 2500 |
| SRR1234505 | Mus musculus | Kidney | 15 | Illumina NextSeq 500 |
| SRR1234506 | Homo sapiens | Blood (PBMC) | 50 | Illumina NovaSeq |
| SRR1234507 | Homo sapiens | Colon tumor | 38 | Illumina HiSeq 4000 |
| SRR1234508 | Drosophila melanogaster | Whole body | 12 | Illumina MiSeq |
| SRR1234509 | Arabidopsis thaliana | Leaf | 18 | Illumina HiSeq 2500 |
| SRR1234510 | Homo sapiens | Pancreatic | 45 | Illumina NovaSeq |

This diversity ensures the model sees different input sizes (12M–50M reads), different organisms (human, mouse, fruit fly, plant), and different sequencing platforms with varying read quality characteristics. Each trace file records per-task runtime, CPU utilization, and peak RSS memory. Trace files are stored in `data/nextflow_traces/`.

### 3.3 WfCommons Traces (90 tasks)

WfCommons [20] is a workflow benchmarking framework that provides recipe-based workflow generators learned from real execution traces. We used the built-in RnaseqRecipe:

```python
from wfcommons.wfchef.recipes import RnaseqRecipe
recipe = RnaseqRecipe.from_num_tasks(100)
workflow = recipe.build_workflow()
```

15 workflow instances with 2–6 samples each were generated in WfFormat JSON [21]. The RnaseqRecipe learns runtime distributions from real nf-core/rnaseq execution profiles, producing traces with realistic structure and relative proportions. JSON files are stored in `data/wfcommons_traces/`.

### 3.4 Synthetic Generator (300 tasks)

We developed a custom generator using lognormal distributions, which are widely used to model computational task runtimes [22]. The stage-specific parameters are listed in Table 7.

**Table 7.** Lognormal distribution parameters for synthetic runtime generation.

| Stage | μ | σ | Median Runtime (20M reads) |
|-------|---|---|---------------------------|
| FastQC | 5.0 | 0.5 | ~150s |
| Trimming | 5.7 | 0.6 | ~300s |
| Alignment | 7.5 | 0.7 | ~1,800s |
| Sort/Index | 6.2 | 0.5 | ~490s |
| MarkDup | 6.5 | 0.6 | ~665s |
| Quant | 5.9 | 0.5 | ~365s |

Formula: `runtime = exp(μ + σ × Z) × √(input_size_factor)`, where Z ~ N(0,1) and input_size_factor = read_count / 20M.

Input sizes sampled from {5M, 10M, 20M, 50M, 100M} reads with ±30% per-sample variation. 50 workflows × 6 stages = 300 tasks. CSV stored in `data/synthetic/`.

## 4. Methodology

### 4.1 Prediction Models

We compared five regression models from scikit-learn [23], summarized in Table 8.

**Table 8.** Regression models compared for runtime prediction.

| Model | Type | Key Property |
|-------|------|-------------|
| Linear Regression | Linear | Simple baseline, assumes straight-line relationship |
| Ridge Regression [24] | Linear + L2 | Shrinks weights to prevent overfitting |
| Lasso Regression [25] | Linear + L1 | Pushes weights to zero (automatic feature selection) |
| SVR (RBF kernel) [26] | Non-linear | Captures curved runtime-input relationship |
| Random Forest [27] | Ensemble | 100 decision trees vote on prediction |

All trained on 360 tasks (80%), evaluated on 90 held-out tasks (20%) with a fixed random seed for reproducibility.

### 4.2 Scheduling Policy: Priority List Scheduling with Round-Up Reservation

We use a priority list scheduler [28] adapted from HEFT [29] for homogeneous processors, combined with a round-up reservation mechanism.

List scheduling [6, 28] maintains a priority-ordered list of ready tasks and assigns the highest-priority task to the next available processor. We use the upward rank priority function from HEFT [29]. Table 9 compares common priority functions used in list scheduling.

**Table 9.** List scheduling priority function variants.

| Variant | Priority Function | Used In |
|---------|------------------|---------|
| FCFS | Arrival order (topological) | Default baseline |
| Shortest Job First | priority = −predicted_runtime | Minimizes response time |
| Longest Job First | priority = predicted_runtime | Reduces idle time |
| **Our policy (upward rank)** | **priority = predicted_runtime + longest predicted downstream path** | **This project** |

Our priority function is the upward rank from HEFT [29]: the task's own predicted runtime plus the longest chain of predicted runtimes from that task to the end of the DAG. This ensures tasks with the most total remaining work get scheduled first.

We add a round-up reservation mechanism to match real SLURM [5] deployment, where each job submission requires a walltime estimate.

**Round-up reservation:** The predicted runtime is rounded up to create a time reservation:

```
reservation(v) = ⌈predicted_runtime(v) / 50⌉ × 50
```

Table 10 illustrates the round-up reservation for several predicted runtimes.

**Table 10.** Round-up reservation examples.

| Predicted | Reserved | Explanation |
|-----------|----------|-------------|
| 245s | 250s | Rounds up to next 50 |
| 251s | 300s | Rounds up to next 50 |
| 1,800s | 1,800s | Already a multiple of 50 |
| 47s | 50s | Minimum one block |

**Scheduling algorithm (non-preemptive):**

1. Find all tasks whose dependencies are satisfied (ready tasks)
2. Compute priority for each ready task using predicted runtimes
3. Select the ready task with the highest priority
4. Assign it to the processor that becomes free earliest
5. Reserve the processor for ⌈predicted_runtime / 50⌉ × 50 seconds
6. Task runs for its TRUE runtime (not predicted)
7. Processor becomes free at: max(true_runtime, reservation)
   - If task finishes early → processor held until reservation expires
   - If task runs over → continues to completion, next task delayed
8. Repeat until all tasks are scheduled

Table 11 summarizes the three scheduling variants compared in our experiments.

**Table 11.** Scheduling variants compared.

| Variant | Priority Source | Reservation |
|---------|---------------|-------------|
| FCFS (No Prediction) | Topological order | None |
| Predicted | ML-predicted runtimes | Round-up of prediction |
| Oracle (Perfect) | True runtimes | Round-up of true runtime |

### 4.3 Experimental Setup

Table 12 lists the experimental configuration.

**Table 12.** Experimental setup.

| Component | Details |
|-----------|---------|
| Language | Python 3.11 (NetworkX [30], scikit-learn [23], NumPy, pandas, matplotlib) |
| Pipeline | nf-core/rnaseq [3] via Nextflow [4] + Conda containers |
| HPC Cluster | University of Kansas (Rocky Linux 9, SLURM [5]) |
| Prediction | 5 regressors, 80/20 split (360 train / 90 test) |
| Workflow Model | RNA-seq DAG: 6 stages per sample + 2 merge tasks |
| Scheduling | Priority list scheduling + round-up reservation (50s blocks) |
| Metric | Makespan (total completion time) |
| Parameters | Samples: 2–12, Processors: 2–10, 10 trials per setting |

## 5. Results

### 5.1 Prediction Accuracy (Finding 1)

SVR achieved the best prediction accuracy among all 5 models, as shown in Table 13. The three linear models (Linear, Ridge, and Lasso) perform nearly identically at around 112% MAPE, indicating that regularization provides no benefit for this feature set. The stage-average baseline, which simply predicts the mean runtime for each stage regardless of input size, scores 102.5% MAPE.

The two non-linear models perform significantly better. Random Forest achieves 68.7% MAPE and SVR achieves the best result at 66.2%, outperforming linear models by approximately 40%. This confirms that the relationship between input size and runtime is non-linear. The RBF kernel in SVR [26] can capture curved scaling patterns, particularly for alignment where genome mapping complexity causes runtime to grow faster than linearly with input size [13].

The R-squared values tell a complementary story. Random Forest has the highest R-squared (0.519) while SVR has a lower value (0.209). This apparent contradiction arises because R-squared measures overall variance explained, while MAPE measures individual prediction accuracy. For scheduling, where each task's predicted runtime directly affects priority ranking, MAPE is the more relevant metric.

**Table 13.** Prediction accuracy on 90 held-out test tasks.

| Model | MAPE (%) | R² |
|-------|----------|-----|
| Linear Regression | ~112 | 0.472 |
| Ridge Regression | ~112 | 0.470 |
| Lasso Regression | ~112 | 0.472 |
| **SVR** | **66.2** | **0.209** |
| Random Forest | 68.7 | 0.519 |
| Stage Average (baseline) | 102.5 | -- |

### 5.2 Prediction Accuracy on Test Set (Finding 2)

Evaluating SVR on the 90 held-out test tasks, prediction accuracy varies by task duration. Short tasks under 500 seconds, which include FastQC, trimming, sorting, and quantification, are predicted relatively well. Longer tasks above 500 seconds, mostly alignment and markdup on larger inputs, show more prediction error. The model captures the general trend that bigger inputs take longer, but exact runtimes are harder to pin down due to inherent variance from genome complexity and I/O patterns.

### 5.3 Error Analysis by Stage (Finding 3)

Using MAE (mean absolute error in seconds) rather than MAPE to avoid misleading results for short tasks, the per-stage errors are presented in Table 14.

**Table 14.** Mean absolute error by pipeline stage (SVR model, 90 test tasks).

| Stage | MAE (seconds) |
|-------|---------------|
| Trimming | 142 |
| Sort/Index | 185 |
| FastQC | 199 |
| Quant | 212 |
| MarkDup | 471 |
| **Alignment** | **2,278** |

Alignment has 5× higher absolute error than any other stage. Its runtimes span a 100× range (from ~100s for small inputs to ~10,000s for large inputs), making it the hardest stage to predict. Alignment is also the longest-running stage and dominates the critical path [29], improving alignment prediction alone would significantly improve scheduling outcomes.

### 5.4 Scheduling Performance (Finding 4)

Table 15 presents the benchmark results with 6 samples, 4 processors, averaged over 10 trials.

**Table 15.** Scheduling performance benchmark.

| Method | Makespan (s) | Improvement over FCFS |
|--------|-------------|----------------------|
| FCFS (no prediction) | 8,230 |, |
| Predicted (SVR + reservation) | 7,957 | +3.3% |
| Oracle (perfect + reservation) | 7,536 | +8.4% |

The prediction-based scheduler captures **39% of the oracle's improvement ceiling**. The oracle saves 694 seconds over FCFS. Our SVR predictions save 273 seconds, 39% of 694. The remaining 61% is lost to prediction errors causing occasional task mis-ranking.

This confirms that the scheduling algorithm (priority list with upward rank [29]) is sound. The gap between predicted and oracle is entirely due to prediction accuracy, better prediction would directly close this gap.

### 5.5 Sensitivity to Workflow Size (Finding 5)

Table 16 shows the scheduling improvement as the number of samples varies, with processors fixed at 4.

**Table 16.** Makespan improvement vs. workflow size (4 processors).

| Samples | Tasks (6n+2) | Improvement |
|---------|--------------|-------------|
| 2 | 14 | ~0% (no contention) |
| 4–8 | 26–50 | 4–5% (sweet spot) |
| 10–12 | 62–74 | ~1% (schedule too packed) |

Prediction helps most at 4–8 samples where scheduling decisions have the most impact. This aligns with scheduling theory: with too few tasks or too many processors, any ordering produces similar results [6].

### 5.6 Improvement Across Sample and Processor Configurations (Finding 6)

Evaluating across combinations of samples (2–10) and processors (2–8), scheduling gains concentrate at 4–8 samples with 2–6 processors, where there is enough contention to create meaningful scheduling choices. At 2 samples or with 8+ processors, there is insufficient contention for reordering to make a difference.

**Sweet spot:** 4–8 samples with 2–6 processors.

### 5.7 Effect of Round-Up Reservation

The round-up reservation (nearest 50 seconds) introduces a tradeoff:

When the predicted runtime exceeds the actual runtime, the processor is held past task completion, wasting time until the reservation expires. When the prediction underestimates, the task runs over its reservation and delays the next task on that processor. The ideal case is when prediction closely matches reality, producing minimal overhead. The 50-second granularity balances two concerns: fine enough to avoid large waste on short tasks, coarse enough to be practical for cluster resource management. This mirrors how SLURM [5] handles walltime reservation in production environments.

## 6. Discussion

One of the clearer findings from this work is that non-linear models are necessary for runtime prediction in RNA-seq workflows. SVR [26] outperforms all three linear models by approximately 40% in MAPE. Linear, Ridge, and Lasso regression all assume that runtime scales proportionally with input size, but this assumption does not hold in practice. Alignment runtime, for instance, grows faster than linearly with read count because genome mapping algorithms like STAR [13] involve index lookups and seed extension steps whose complexity depends on both the number of reads and the repetitiveness of the genome. SVR with an RBF kernel can model these curved relationships, which explains its advantage.

Alignment emerges as the central bottleneck in two distinct ways. It is the longest-running stage in the pipeline, typically taking around 30 minutes per sample and dominating the critical path of the workflow DAG. It is also the hardest stage to predict, with a mean absolute error of 2,278 seconds compared to less than 500 seconds for every other stage. This dual role means that improvements in alignment prediction would have an outsized effect on scheduling quality. If alignment predictions were even 20% more accurate, the scheduler would make better priority decisions on exactly the tasks that matter most for makespan.

The comparison between predicted and oracle scheduling reveals that prediction accuracy, not the scheduling algorithm, is the limiting factor. The oracle achieves an 8.4% makespan reduction using the same priority list scheduling policy as our predicted scheduler. Our SVR-based predictions capture only 39% of that ceiling. The scheduling algorithm itself, priority list scheduling with upward rank [28, 29], is well-suited to the problem. The gap between 39% and 100% of the oracle is entirely due to prediction errors that cause the scheduler to occasionally assign the wrong priority to tasks.



## 7. Future Work

The most immediate improvement would be collecting more real training data. Only 60 of our 450 tasks come from real Nextflow executions, and increasing this proportion would improve prediction generalization to unseen workloads.

Adding XGBoost [31] to the model comparison is another natural next step. Gradient boosting builds trees sequentially, where each new tree corrects the errors of the previous ones. This approach often outperforms both SVR and Random Forest on tabular regression tasks.

Extending the scheduler to heterogeneous processors would increase the potential for improvement. With processors of different speeds, the scheduler must decide not only which task goes first but also which processor it should run on. This is the full HEFT [29] problem, where prediction accuracy becomes even more critical.

Uncertainty-aware scheduling is a longer-term direction. Instead of producing a single predicted runtime, the model could output a confidence interval. The scheduler would then set wider reservation buffers for uncertain predictions and tighter ones for confident predictions, reducing both wasted time and run-over risk.

Additional input features such as read length, genome size [13], and hardware-specific metrics could reduce prediction error across all stages. Currently the model only sees stage type and input size, which limits its ability to distinguish between tasks with the same read count but different complexity characteristics.

Finally, the round-up granularity could be made dynamic rather than fixed at 50 seconds. Short tasks like FastQC could use finer blocks (10 seconds) while long tasks like alignment could use coarser blocks (100 seconds), reducing wasted reservation time without sacrificing scheduling practicality.

## 8. References

[1] Wang, Z., Gerstein, M., & Snyder, M. (2009). RNA-Seq: a revolutionary tool for transcriptomics. Nature Reviews Genetics, 10(1), 57–63.

[2] Conesa, A., et al. (2016). A survey of best practices for RNA-seq data analysis. Genome Biology, 17(1), 13.

[3] Ewels, P. A., Peltzer, A., Fillinger, S., et al. (2020). The nf-core framework for community-curated bioinformatics pipelines. Nature Biotechnology, 38(3), 276–278.

[4] Di Tommaso, P., Chatzou, M., Floden, E. W., et al. (2017). Nextflow enables reproducible computational workflows. Nature Biotechnology, 35(4), 316–319.

[5] Yoo, A. B., Jette, M. A., & Grondona, M. (2003). SLURM: Simple Linux Utility for Resource Management. In Job Scheduling Strategies for Parallel Processing (JSSPP), Springer, 44–60.

[6] Graham, R. L. (1969). Bounds on multiprocessing timing anomalies. SIAM Journal on Applied Mathematics, 17(2), 416–429.

[7] Tsafrir, D., Etsion, Y., & Feitelson, D. G. (2007). Backfilling using system-generated predictions rather than user runtime estimates. IEEE TPDS, 18(6), 789–803.

[8] Bader, J., Lehmann, F., Thamsen, L., Kao, O., & Weidlich, M. (2023). Lotaru: Locally Estimating Runtimes of Scientific Workflow Tasks in Heterogeneous Clusters. SSDBM.

[9] Hilman, M. H., Rodriguez, M. A., & Buyya, R. (2018). Task runtime prediction in scientific workflows using an online incremental learning approach. IEEE/ACM International Conference on Utility and Cloud Computing (UCC), 93–102.

[10] Da Silva, R. F., et al. (2017). A characterization of workflow management systems for extreme-scale applications. Future Generation Computer Systems, 75, 228–238.

[11] Andrews, S. (2010). FastQC: a quality control tool for high throughput sequence data. Babraham Bioinformatics.

[12] Krueger, F. (2015). Trim Galore: a wrapper around Cutadapt and FastQC. Babraham Bioinformatics.

[13] Dobin, A., et al. (2013). STAR: ultrafast universal RNA-seq aligner. Bioinformatics, 29(1), 15–21.

[14] Li, H., et al. (2009). The Sequence Alignment/Map format and SAMtools. Bioinformatics, 25(16), 2078–2079.

[15] Broad Institute. (2019). Picard toolkit. http://broadinstitute.github.io/picard/

[16] Patro, R., Duggal, G., Love, M. I., Irizarry, R. A., & Kingsford, C. (2017). Salmon provides fast and bias-aware quantification of transcript expression. Nature Methods, 14(4), 417–419.

[17] Ewels, P., Magnusson, M., Lundin, S., & Käller, M. (2016). MultiQC: summarize analysis results for multiple tools. Bioinformatics, 32(19), 3047–3048.

[18] Love, M. I., Huber, W., & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology, 15(12), 550.

[19] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.

[20] Coleman, T., et al. (2022). WfCommons: A Framework for Enabling Scientific Workflow Research and Development. Future Generation Computer Systems, 128, 16–27.

[21] WfCommons. WfFormat Specification. https://github.com/wfcommons/WfFormat

[22] Feitelson, D. G. (2015). Workload Modeling for Computer Systems Performance Evaluation. Cambridge University Press.

[23] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

[24] Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67.

[25] Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. Journal of the Royal Statistical Society B, 58(1), 267–288.

[26] Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1996). Support vector regression machines. NeurIPS, 9.

[27] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

[28] Kwok, Y. K., & Ahmad, I. (1999). Static scheduling algorithms for allocating directed task graphs to multiprocessors. ACM Computing Surveys, 31(4), 406–471.

[29] Topcuoglu, H., Hariri, S., & Wu, M. (2002). Performance-effective and low-complexity task scheduling for heterogeneous computing. IEEE TPDS, 13(3), 260–274.

[30] Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. SciPy, 11–15.

[31] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD, 785–794.

## Appendix A: Links and Resources

**Table 17.** Project links and resources.

| Resource | URL |
|----------|-----|
| nf-core/rnaseq pipeline | https://nf-co.re/rnaseq |
| nf-core/rnaseq source | https://github.com/nf-core/rnaseq |
| Nextflow | https://www.nextflow.io/ |
| WfCommons | https://wfcommons.org/ |
| WfCommons GitHub | https://github.com/wfcommons/wfcommons |
| WfFormat specification | https://github.com/wfcommons/WfFormat |
| Sequence Read Archive (SRA) | https://www.ncbi.nlm.nih.gov/sra |
| KU HPC documentation | https://hpc.ku.edu/ |
| scikit-learn | https://scikit-learn.org/ |
| Project repository | [your git link] |

## Appendix B: Declaration of AI Usage

I used AI tools to assist with writing Python simulation code, generating visualization scripts, and refining the language of this report. All research decisions, including the choice of RNA-seq domain, data collection strategy, model selection, scheduling policy design, and interpretation of results, were made by me independently. The data collection on KU's HPC cluster (Nextflow setup, Conda configuration, SLURM job submission) was done by me. AI served as a coding and writing tool; the research contributions are my own.
