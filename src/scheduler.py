"""FCFS and priority list scheduling with round-up reservation."""
import math
import networkx as nx


def schedule_fcfs(G, info, n_proc):
    """Baseline: topological order, no reservation."""
    pfa = [0.0] * n_proc
    ft = {}
    for task in nx.topological_sort(G):
        dep = max((ft[p] for p in G.predecessors(task)), default=0.0)
        bp = min(range(n_proc), key=lambda p: max(pfa[p], dep))
        ft[task] = max(pfa[bp], dep) + info[task]["runtime"]
        pfa[bp] = ft[task]
    return max(ft.values())


def schedule_predicted(G, info, estimates, n_proc, round_unit=50):
    """Priority list scheduling with round-up reservation.
    
    Priority = upward rank (predicted runtime + longest downstream path).
    Reservation = ceil(predicted / round_unit) * round_unit.
    Processor held for max(true_runtime, reservation).
    """
    def ru(t):
        return math.ceil(t / round_unit) * round_unit

    # compute upward rank
    pri = {}
    for task in reversed(list(nx.topological_sort(G))):
        children = list(G.successors(task))
        pri[task] = estimates[task] + (max(pri[c] for c in children) if children else 0)

    pfa = [0.0] * n_proc
    ft = {}
    rem = set(G.nodes())

    while rem:
        ready = [t for t in rem if all(p in ft for p in G.predecessors(t))]
        if not ready:
            break
        ready.sort(key=lambda t: -pri[t])
        ch = ready[0]
        rem.remove(ch)

        dep = max((ft[p] for p in G.predecessors(ch)), default=0.0)
        bp = min(range(n_proc), key=lambda p: max(pfa[p], dep))
        start = max(pfa[bp], dep)

        true_rt = info[ch]["runtime"]
        reserved = ru(estimates[ch])
        ft[ch] = start + true_rt
        pfa[bp] = start + max(true_rt, reserved)

    return max(ft.values())
