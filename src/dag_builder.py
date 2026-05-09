"""Build RNA-seq workflow DAGs with NetworkX."""
import networkx as nx
from utils import STAGES


def build_dag(n_samples, runtimes):
    """Build DAG for n samples. runtimes maps task_id -> seconds."""
    G = nx.DiGraph()
    info = {}
    quant_ids = []

    for s in range(n_samples):
        nd = {}
        for stage in STAGES:
            tid = f"S{s}_{stage}"
            G.add_node(tid)
            info[tid] = {"stage": stage, "runtime": runtimes[tid]}
            nd[stage] = tid

        G.add_edge(nd["fastqc"], nd["trimming"])
        G.add_edge(nd["trimming"], nd["alignment"])
        G.add_edge(nd["alignment"], nd["sort_index"])
        G.add_edge(nd["alignment"], nd["markdup"])
        G.add_edge(nd["sort_index"], nd["quant"])
        G.add_edge(nd["markdup"], nd["quant"])
        quant_ids.append(nd["quant"])

    # merge tasks
    for ms in ["multiqc", "deseq2"]:
        mid = f"MERGE_{ms}"
        G.add_node(mid)
        info[mid] = {"stage": ms, "runtime": runtimes.get(mid, 60.0)}
        for q in quant_ids:
            G.add_edge(q, mid)

    return G, info
