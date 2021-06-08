"""
custom math functions

Created: Monday, ‎June ‎7, ‎2021, ‏‎6:38:14 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import numpy as np


def is_pos_int(x):
    """ check if x is a positive integer """
    if not isinstance(x, int):
        return False
    elif x < 1:
        return False
    else:
        return True


def get_factors(x, neg=False, ub=None):
    """
    find all factors of integer x; not very efficient

    Args:
    -----
    neg (bool): if True, return negative factors
    ub (positive int): if present, keep only factors whose abs val <= ub
    """
    if not is_pos_int(x):
        raise TypeError("<x> has to be a positive integer")
    if x == 1:
        return [1]
    factors = []
    q = x // 2  # q is mid-point (even x) or 0.5 less than mid-point (odd x)
    range_max = q + 1
    if ub is not None:
        if not is_pos_int(ub):
            raise TypeError("<ub> has to be a positive integer")
        else:
            range_max = min(range_max, ub + 1)
    if not neg:
        for i in range(1, range_max):
            if x % i == 0:
                factors.append(i)
        factors.append(x)
    else:
        for i in range(1, range_max):
            if x % i == 0:
                factors.append(-i)
        factors.append(-x)
    if ub is not None:
        if ub < x:
            factors.pop()
    return factors


def step_funct(val_min, val_max, num, steps_tot):
    """generate a list of vals (for discrete step functions)

    Args:
    -----
    num (int): number of steps, each with a unique val
    steps_tot (int): number of total steps;
        has to be int multiple of <num> (i.e., <num> has to be a factor of <steps_tot>)

    Intermidiate:
    -------------
    if num=1, only return [val_min] * steps_tot
    else:
        rep = steps_tot // num (quotient)
        d = (val_max - val_min) / (num - 1) # returned by np.linspace(retstep = True)

    Return:
    -------
    arr (np.arr): [val_min, val_min + d, ..., val_min + (num - 1) * d]
    where each entry is repeated <rep> times
    """
    if not is_pos_int(num):
        raise TypeError("<num> has to be a positive integer")
    arr = np.linspace(val_min, val_max, num, endpoint=True)
    rep, rep_last = divmod(steps_tot, num)
    if rep_last != 0:  # <steps_tot> is not int multiple of <num>
        raise NotImplementedError(f"<steps_tot>={steps_tot} is not int multiple of <num>={num}")

    return np.repeat(arr, [rep] * num)


def findNearest(arr, val, is_arg=True):
    """
    Arg:
        arr (any list like object)
        val (any number): the value to which one wants to find in arr that is nearest
        is_arg (bool): if False, return the value instead of argument/index
    Return:
        index or value depending on is_arg
    """
    arr_ = np.array(arr)  # convert to nparr, make a copy by default
    ind = np.abs(arr_ - val).argmin()
    if is_arg:
        return ind
    else:
        return arr[ind]


# region: linear algebra

"""
    ########################################
        W_norm() was originally from NetworkScience.py
        but it was modifed here.
    ########################################
"""


def W_norm(W, axis=1, filler=None):  # updated on 2021.3.24, much simpler this time
    """
    Args:
    -----
    W (np.arr): undirected/directed Weight Matrix, self-loop matters, be cautious
        when "completely isolated", the vertex in W have self-loop value of 0 and degree 0
        not completely isolated if self-loop!=0 even if the edges to other vertices have 0 weight
    axis (0/1): normalize W along columns/rows (by default 1 because i->j: W[i,j])
    filler: value to replace nan when vertex is completely isolated
    """
    normed = W / np.sum(W, axis, keepdims=True)
    # if completely isolated, will produce a row (if axis=1)/column (if axis=0) of nan
    if filler is not None:
        normed[normed == np.nan] = filler
    return normed


def A2P(A, axis=1):  # input count matrix, output transition probability
    denom = np.sum(A, axis, keepdims=True)  # sum across column in any sample
    return np.divide(A, denom, where=(denom != 0))  # if there is no count in that row, keep 0


def getUpperTriangle(W, diag=True, up=True):
    """
    Arg:
        W (np.arr; (n,n)): if not square matrix, raise ValueError
        diag (bool): if False, then exclude diagonals (self-loop)
        up (bool): if False, then use lower triangle instead
    Return:
        (Flattened list): e.g., 1,2,3,4,5,6...
        (including diagonals)
        up: 1 2 3   down: 1
              4 5         2 4
                6         3 5 6
    """
    nrow, ncol = np.shape(W)
    if nrow != ncol:
        raise ValueError("<W> is not square.")
    if up:
        return [e for list_ in [W[i, i + int(not diag) :] for i in range(nrow)] for e in list_]
    else:
        return [e for list_ in [W[i + int(not diag) :, i] for i in range(nrow)] for e in list_]


def rank_eigvals(A, eig_k=None, rtol=1e-05):
    """
    Return:
        eigtup = (eigvals, eigvecs, l, kth_eigval, kth_eigvec)
        ~ or ~
        eigtup = (eigvals, eigvecs, l)
        l: ranking of the eigvals (e.g., eigvals=[1,0.7,0.7,0.6] -> [0,1,1,2])
    """
    # per (numpy v1.19) https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh
    # "The eigenvalues in ascending order, each repeated according to its multiplicity."
    np.linalg.eigh(A)
    """ ↑ ❗ if I run this twice, the 2nd time it will be fine;
    otherwise will not function if ~ draw 2-panel plot ~ is ran in plotGraph()"""
    eigvals, eigvecs = np.linalg.eigh(
        A
    )  # eigenvalues could be repeated (especially for symmetric Graph)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]  # descending order
    n = len(eigvals)
    l = [0 for i in range(n)]
    start_idx = 0  # starting index of the repeated value
    last_rank = 0  # self-explanatory
    for i in range(1, n):
        if eigvals[i] < eigvals[i - 1] * (1 - rtol):
            l[start_idx:i] = [last_rank] * (i - start_idx)
            start_idx = i  # update starting index after finishing with last largest value
            last_rank += 1  # update rank to be the current one
    l[start_idx:] = [last_rank] * (n - start_idx)

    if eig_k is not None:
        try:
            idx = l.index(eig_k)  # first one that matches
        except:  # can't find eig_k
            print("Can't find eig_k.")
            print("is A symmetric? ", np.allclose(A, A.T, rtol=1e-05, atol=1e-08))
            idx = 0
        kth_eigval = eigvals[idx]
        kth_eigvec = eigvecs[:, idx]  # nparr; ↓ min-max normalization ↓
        # kth_eigvec = (kth_eigvec - kth_eigvec.min()) / (kth_eigvec.max() - kth_eigvec.min())
        return (eigvals, eigvecs, l, kth_eigval, kth_eigvec)
    else:
        return (eigvals, eigvecs, l)


# endregion
