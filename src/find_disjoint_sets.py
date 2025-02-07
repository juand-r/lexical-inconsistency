import random
random.seed(42)

import numpy as np
np.random.seed(42)

from networkx.algorithms.community import kernighan_lin_bisection

#S, T = kernighan_lin_bisection(G, partition=None, seed=0, max_iter=10)
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection


def build_bipartite_graph(items):
    G = nx.Graph()
    for obj in items:
        G.add_node(obj.noun1)
        G.add_node(obj.noun2)
        G.add_edge(obj.noun1, obj.noun2, weight=1)
    return G


def deterministic_kernighan_lin(G):
    # Suppose we pick a stable, deterministic initial partition:
    nodes = sorted(G.nodes())   # sorted for consistent ordering
    half = len(nodes) // 2
    # First half in one set, second half in the other, for example:
    partition_init = (set(nodes[:half]), set(nodes[half:]))

    S, T = kernighan_lin_bisection(
        G,
        partition=partition_init,  # no randomness here
        max_iter=10,
        seed=0                     # ensures tie-breaks are consistent
    )
    return S, T


def partition_items_kernighan_lin(items):
    G = build_bipartite_graph(items)

    # We run Kernighan-Lin, which returns a tuple of two node sets (S, T).
    # It tries to minimize edge cuts while dividing the graph into two sets.
    S, T = deterministic_kernighan_lin(G)#, max_iter=10, weight='weight')

    # Build cluster1 (edges whose endpoints are in S) and cluster2 (endpoints in T).
    cluster1 = []
    cluster2 = []
    excluded = []
    for obj in items:
        in_S_1 = obj.noun1 in S
        in_S_2 = obj.noun2 in S
        if in_S_1 and in_S_2:
            cluster1.append(obj)
        elif not in_S_1 and not in_S_2:
            cluster2.append(obj)
        else:
            # crosses between S and T -> we cannot keep it if we want disjoint sets
            excluded.append(obj)

    return cluster1, cluster2, excluded



