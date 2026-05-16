"""
Visualize community in Sierpiński graphs
both defined community and detected communities
this script is so far unused (as of 2021.10.25)
Created: Thursday, ‎October ‎14, ‎2021, ‏‎10:06:10 AM (EDT)
@author: Xiaohuan (Pixel) X.
"""

from itertools import product  # for nested loops
import igraph as ig

from utility427.helper427 import set_dir427, mkdir_p, get_params, partial_427_decorator
from utility427.plt427 import plt, Normalize, LinearSegmentedColormap, GridSpec, Line2D  # mpl
from utility427.plt427 import saveNclose427, colors_selector, cbrLabel427  # mpl helpers
from utility427.Sierpinski427 import make_Sierpinski427, p2ten, p_ary, make_SierpinskiGraph427


def main():
    """
    PLACEHOLDER
    """
    tups = [[0, 1, 3], [3, 4, 5], [3, 4, 5]]  # regType, p(power), n(level) = tup
    # dd = make_gt_sier(tups)
    assignments = get_assignments([i for i in range(round(3 ** 3 + 1))], regType=1, p=3, n=3)
    print(f"DEBUG assignments: {assignments}")


def get_assignments(nodeidx, regType, p, n):
    """return community assignment for each level in Sierpiński

    for regType=0 case:
    l-module, where l = 1,...,n (1 is finest; if l=n, it's the whole graph, a trivial assignment)
    denote p_ary of idx by pstr
    then leading n-l digits (pstr[0:n-l]) are module index
    each containing p^l nodes
        they share the same pstr[0:n-l]
        and are assignmed module index pstr[0:n-l]
    hence assignments[l] = [p2ten(pstr[0:n-l], p=p) for pstr in nodeidx_p]

    each l-module typically contains mixed level edges:
    each 1-module contains level-1 edges
    each 2-module contains level-1, level-2 edges
    ...
    each n-module contains level-1, level-2, ..., level-n edges


    Args
    ----
    - nodeidx (list): list of node indices (int; 0,1,...)
    - regType (int):
        0,3: no reg or self-loop reg; both share the same rule
        1: one-node reg; l = 1,...,n,n+1
        the last node never joins modules when l++
        even when l=n, we have 2 modules; l=n+1 is trivial assignment

    Return
    ------
    - assignments (dict): key:val -> level:list of int (base-10) assignment (i.e., module index)
    """
    # find p_ary for the base10 nodeidx
    assignments = dict()

    if regType in [0, 3]:
        nodeidx_p = [p_ary(x, p=p, L=n) for x in nodeidx]
        for l in range(1, n):
            assignments[l] = [p2ten(pstr[0 : n - l], p=p) for pstr in nodeidx_p]
        assignments[n] = [0 for _ in nodeidx_p]
    elif regType in [1]:
        nodeidx_p = [p_ary(x, p=p, L=n + 1) for x in nodeidx]
        # essentially the new n is n=n+1
        for l in range(1, n + 1):
            assignments[l] = [p2ten(pstr[0 : n - l + 1], p=p) for pstr in nodeidx_p]
        assignments[n + 1] = [0 for _ in nodeidx_p]

    return nodeidx_p, assignments


def vis_weight(ws):
    """
    visualize weight matrices
    """
    pass


def make_gt_sier(tups):
    """first generate a bunch of ground truths (i.e., beta=∞)

    Return
    ------
    - dd (dict): DataDict = {(regType,p,lv):{beta:SierDict}}
        where SierDict is dict returned by make_SierpinskiGraph427
        beta = -1 means infinity (i.e., ground truth)

    """
    dd = dict()
    for tup in product(*tups):
        regType, p, n = tup  # unpack
        dd[tup] = dict()
        dd[tup][-1] = make_SierpinskiGraph427(p, n, norm=True, regType=regType, use_set=False)

    return dd


if __name__ == "__main__":
    main()
