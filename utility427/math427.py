"""
custom math functions

Created: Monday, ‎June ‎7, ‎2021, ‏‎6:38:14 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import numpy as np


def log_b(x, b):  # return base b logarithm of x
    if b <= 0:
        raise ValueError("base <b> has to be strictly positive")
    return np.log(x) / np.log(b)


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

    Args
    ----
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


def step_funct(num, steps_tot, val_min=None, val_max=None):
    """generate a list of vals (for discrete monotonic step functions)

    Args
    ----
    - num (int or 1D-list or np.arr):
        - int: number of steps, each with a unique val
        - 1D-list or np.arr: ignore val_min, val_max
            instead, use vals in num; len(num) = number of steps
    - steps_tot (int): number of total steps;
        has to be int multiple of <num> (i.e., <num> has to be a factor of <steps_tot>)

    Intermediary
    ------------
    if num=1, only return [val_min] * steps_tot
    else:
        rep = steps_tot // num (quotient)
        d = (val_max - val_min) / (num - 1) # returned by np.linspace(retstep = True)

    Return
    ------
    - arr (np.arr): [val_min, val_min + d, ..., val_min + (num - 1) * d]
        where each entry is repeated <rep> times
    """
    if is_pos_int(num):
        if val_min is None or val_max is None:
            raise Exception("both <val_min> and <val_max> have to be supplied")
        arr = np.linspace(val_min, val_max, num, endpoint=True)
        rep, rep_last = divmod(steps_tot, num)
        if rep_last != 0:  # <steps_tot> is not int multiple of <num>
            raise NotImplementedError(f"<steps_tot>={steps_tot} is not int multiple of <num>={num}")

        return np.repeat(arr, [rep] * num)
    elif isinstance(num, (list, np.ndarray)):
        rep, rep_last = divmod(steps_tot, len(num))
        if rep_last != 0:  # <steps_tot> is not int multiple of <num>
            msg = f"<steps_tot>={steps_tot} is not int multiple of len(<num>)={len(num)}"
            raise NotImplementedError(msg)

        return np.repeat(num, [rep] * len(num))
    else:
        raise TypeError("<num> has to be a positive integer, 1D-list, or np.arr")


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


def func1d_bs427(arr_1d, funct_stat0, rng_idx, funct_stat1=None, repeat=False):
    """actual bootstrap happens here
    NOTE: len(arr_1d) == len(rng_idx[i]) for any i should be true

    Args
    ----
    - arr_1d (nparr): has to be 1D
    - funct_stat (function: 1D->scalar): np.nanmedian and the like; statistic used in bootstrapping
    - rng_idx (list): list of np.arr which is indices for each bootstrap resample
    - funct_stat1 (function: 1D->scalar): np.nanmedian and the like;
        - statistic (funct_stat1) of statistic (funct_stat)
        - if not None,
            - val=funct_stat1(arr) will be repeated len(arr_1d) times if repeat==True
            - otherwise val will be return (scalar)

    Return
    ------
    (list): 1D, size=len(rng_idx) if stat or len(arr_1d) if stat2
    """

    arr = [None] * len(rng_idx)
    for i in range(len(rng_idx)):
        arr[i] = funct_stat0(arr_1d[rng_idx[i]])
    if funct_stat1 is not None:
        arr = [funct_stat1(arr)] * len(arr_1d) if repeat else funct_stat1(arr)

    return arr


def bootstrap427(nparr, axis, n_sample, statistic0, statistic1=None, repeat=False):
    """bootstrap is same-size sample w replacement

    Assume axes other than `axis` were independently sampled;
    this way we only need to RNG `n_sample` times
    e.g., nparr has 4 dims; boostrap axis=3; then nparr[s,l,b,:] is independent of
        nparr[s+1,l,b,:], nparr[s,l+1,b,:], nparr[s,l,b+1,:]

    Args
    ----
    - nparr (np.arr): data
    - axis (int): axis to bootstrap
    - n_sample (int): number of samples to bootstrap
    - statistic0 (str): statistic to calculate per bootstrap resample; output should be a scalar
    - statistic1 (None or str): statistic of statistic
        - should be "std": bootstrap to get the "spread" of the statistic of interest

    Return
    ------
    - (np.arr):
        - case 1: statistic1 is None
        same shape as nparr except arr[`axis`]=`n_sample` instead of nparr.shape[`axis`];
        extry along axis=`axis` is statistic at each i in range(`n_sample`)
        - case 2: statistic1 is not None
        same shape as nparr;
        entry along axis=`axis` is repeated value of the mean/std/etc of the statistic
    """

    n = nparr.shape[axis]  # number of sample in the data at given <axis>
    rng_idx = [np.random.randint(n, size=n) for _ in range(n_sample)]  # initialize random index

    kwargs = dict(rng_idx=rng_idx, repeat=repeat)
    for i, func1d in enumerate([statistic0, statistic1]):
        if i==1 and func1d is None:
            kwargs[f"funct_stat{i}"] = None
        elif func1d == "mean":
            kwargs[f"funct_stat{i}"] = np.nanmean
        elif func1d == "std":
            kwargs[f"funct_stat{i}"] = np.nanstd
        elif func1d == "median":
            kwargs[f"funct_stat{i}"] = np.nanmedian
        else:
            raise NotImplementedError(f"<func1d>={func1d} is unimplemented")

    return np.apply_along_axis(func1d_bs427, axis=axis, arr=nparr, **kwargs)


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
