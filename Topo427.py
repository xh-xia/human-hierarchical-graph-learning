"""
Graph Topological measures on Sierpiński graph
Created: Monday, ‎June ‎21, ‎2021, ‏‎9:16:26 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

from itertools import product  # for different graph parameters
import igraph

from utility427.Sierpinski427 import make_SierpinskiGraph427


def main():
    hierList = [[3, 0, 1, 2, 3], [3, 4, 5], [3, 4, 5]]
    dd_list = map(do_Sier_A, product(*hierList))
    DD = dict()
    for dd in dd_list:  # aggregate all keys of dd in dd_list into DD
        DD.update(dd)
    print(f"DD:\n {DD.keys()}")


def do_Sier_A(tup):
    """
    1. generate GTDict for Sierpiński graph of parameter set `tup`
    2. process the graph with some topological measures

    Intermediary
    ------------
    the keys in dd[tup] are:
    - "A", "edgeList", "lvList", "n": generated from make_SierpinskiGraph427()
    - "nm_<alias>": node measure with alias of the measure
    - "em_<alias>": edge measure with alias of the measure
    - "gm_<alias>": graph/global measure with alias of the measure

    Structural Properties of Graphs (python-igraph):
    c (comprehensive): https://igraph.org/c/doc/igraph-Structural.html
    python: https://igraph.org/python/doc/tutorial/tutorial.html#structural-properties-of-graphs
        Graph object methods: https://igraph.org/python/doc/api/igraph.Graph.html
    we create it as a directed graph, but from a symmetric adjacency matrix
    - deg: degree
    - NBC: Node Betweenness Centrality
    - EBC: Edge Betweenness Centrality
    - LCC: Local Clustering Coefficient (only for undirected simple (no self-loops) graph)
    """
    regType, p, lv = tup  # unpack tup
    print(tup)
    dd = dict()  # = DataDict = {(regType,p,lv):{"A": A,"edgeList": edgeList, etc.}}
    dd[tup] = dict()
    dd[tup] = make_SierpinskiGraph427(p, lv, norm=False, regType=regType)
    dd[tup]["igraph"] = igraph.Graph.Adjacency(dd[tup]["A"].tolist(), mode="directed")
    N = dd[tup]["igraph"].vcount()  # num of nodes
    E = dd[tup]["igraph"].ecount()  # num of edges (double count for directG; self-loop is 1)

    dd[tup]["nm_deg"] = dd[tup]["igraph"].degree(mode="out")  # if "all", double count for directG
    dd[tup]["nm_NBC"] = dd[tup]["igraph"].betweenness()  # unnormalized; R ver. has norm option
    dd[tup]["nm_NBC"] = [x / ((N - 1) * (N - 2)) for x in dd[tup]["nm_NBC"]]  # normalized
    dd[tup]["em_EBC"] = dd[tup]["igraph"].edge_betweenness()  # unnormalized
    dd[tup]["em_EBC"] = [x / (N * (N - 1)) for x in dd[tup]["em_EBC"]]  # normalized
    # dd[tup]["nm_LCC"] = dd[tup]["igraph"].transitivity_local_undirected(mode="nan")

    temp = dd[tup]["nm_NBC"]
    temp = set(temp)
    print(f"DEBUG E:\n{temp}")
    exit(427)
    return dd


if __name__ == "__main__":
    main()