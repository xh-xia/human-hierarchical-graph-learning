"""
parameter related functions for simulation (sim427 folder)
dependency (local): helper427.py
NOTE: because all utility427 scripts will be run imported, sys.path won't have their script dir
      so we need to add them (if running the script directly, this step is effectively redundant)
      we insert them to the start of sys.path,
      making sure the import afterwards will search the script path first
Created: Monday, ‎June ‎7, ‎2021, ‏‎3:27:52 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""

import numpy as np
from itertools import product, chain

import sys, os, inspect
temp_cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, temp_cwd)
from helper427 import unique_iter
from math427 import get_factors
sys.path.pop(0)  # remove script dir from sys.path


def make_beta_mat(N, n, binned=False, b=10, seed=427):
    """
    it uses its own seed for uniform sampling; separated from the simulation

    Return
    ------
    - beta_mat (2D nparr): dim=(N, n+1); for i = 0,...,N-1 | j = 0,...,n-1:
        beta_mat[i,j]: actual beta for agent j centered at beta=beta_mat[i,-1]
        each beta_mat[i,j] is a point uniformly sampled from logspace of beta
        the center of this logspace is at beta=beta_mat[i,-1]
    """
    beta_mat = np.zeros((N, n + 1), dtype=float)
    # first generate log-uniformly spaced beta_arrs of len=N from 10**-3 to 10**1
    # aim to cover both end points but due to floating point error the last point may not be 10**1
    t0, t1 = -3, 1
    dt = (t1 - t0 ) / (N - 1)  # (linearly) evenly spaced
    if not binned:
        for i in range(N):
            beta_mat[i, :] = b ** (t0 + dt * i)
    else:
        RNG = np.random.default_rng(seed=seed)
        rand01 = RNG.random((N, n))
        t0 -= dt  # extend the start and end with two more points
        for i in range(N):
            beta_mat[i, -1] = b ** (t0 + dt * (i + 1))
            aa = t0 + dt * (i + 1 - 0.5)
            bb = t0 + dt * (i + 1 + 0.5)
            beta_mat[i, :-1] = b ** ((bb - aa) * rand01[i, :] + aa)
    return beta_mat


def make_sim_params(params):
    """set up parameters for simulations

    Arg - params (dict)
    -------------------
    NOTE: params should be read from params.json file, which should be pure json w/o comments
    - key_classes (list): list of key_class (not implemented)
        - key_class (str): type of simulation run; encoding regType, p, n parameters
            e.g., "reg_n_p", "r"
    - n_agents (int): num of agents per param (including beta)
    - n_beta_constcase (int): num of beta in constant beta case
    - SEED (int): seed for current simulation params
    - steps_tot (int): total number of steps for random walk
    - sample_period (int): sample sim results every <sample_period> steps
    - var_betas (nested list of bool or 1D-list or np.arr):
        NOTE: it's not generic list-like or arr-like
            bool: whether we extend codenames (variable beta case) in "beta_arrs"[i]
                if True, we will use params_max_beta.json to find specific beta val
            1D-list/np.arr: current deprecated
                must contain only two numbers: val_min, val_max in step_funct()
                for bool: val_min, val_max = 10 ** (-3), 10 ** (1)
    - ub (int): upperbound for variable beta (shows up in get_factors())

    Intermediary
    ------------
    - params["pd"] (iter): reg, n, p tuple iterator
    - params["int_max"] (int32): some constant, in this case 2147483647 for int (int32 really)
    - params["beta_acts"] (list of 2D np.arr): actual beta list
    - params["beta_arrs"] (list of 1D np.arr): group beta list
     entry in the list e.g., [0.01, 0.1, 1,| -1, -2, -3]
                            (constant case)|(variable case)
        1D arr of beta values (constant beta case)
        also incorporates variable beta case (monotonic step function w/ certain <sf_width>)
        sweeping from beta_min to beta_max (if beta_max < beta_min, it's decreasing)
            <0 val (comes after constant case) in the arr is only a codename;
            denote <0 val as -x (where x is positive int):
            x is number of times beta changes (e.g., x=1, beta takes 2 vals, changed once)
            e.g., vals = []
                beta_increment = (beta_max - beta_min) / x
                beta (at sf_step i, i=0,1,...,x)
                = beta_min + beta_increment * i
                sf_width (num of steps in each sf_step; these steps have same beta)
                = steps_tot / (x + 1)
                (make sure steps_tot % (x + 1) = 0)
                    steps at sf_step = 0: 0,1,...,sf_width-1
            NOTE: since x has to be positive integer, put alongside float beta (const case),
                  the negative beta (var case) will be float as well;
                  so when extracting codenames (var case), use round(-x) to get original int first
    - params["beta_classes"] (list of str): type of beta
        - "constant"
        - "step_max_beta"
        - "step_{}to{}".format(*params["var_betas"][i])
    - params["range_agents"] (range object): range(params["n_agents"])
    """
    params["int_max"] = np.iinfo(int).max

    # beta and beta classes (they have the same length)
    params["beta_arrs"] = [None] * len(params["var_betas"])  # initialize group beta list
    params["beta_acts"] = [None] * len(params["var_betas"])  # initialize actual beta list
    params["beta_classes"] = [None] * len(params["var_betas"])  # initialize beta class list
    for i in range(len(params["var_betas"])):
        # for each item in params["var_betas"],
        # it has a corresponding item in beta_arrs and beta_classes
        beta_mat = make_beta_mat(params["n_beta_constcase"], params["n_agents"], binned=True)
        params["beta_arrs"][i] = beta_mat[:, -1]  # group beta
        params["beta_acts"][i] = beta_mat[:, :-1]  # actual beta
        params["beta_classes"][i] = "constant_binned"
        if isinstance(params["var_betas"][i], bool) and params["var_betas"][i]:
            params["beta_classes"][i] = "step_max_beta"
            params["beta_arrs"][i] = np.concatenate([params["beta_arrs"][i], [-427]], axis=None)
        elif isinstance(params["var_betas"][i], (list, np.ndarray)):
            raise Exception(f"<params['var_betas][{i}]> being non-bool is currently deprecated")
            # kw_factors = dict(neg=True, ub=params["ub"])
            # params["beta_classes"][i] = "step_{}to{}".format(*params["var_betas"][i])
            # params["beta_arrs"][i] = np.concatenate(
            #     [params["beta_arrs"][i], get_factors(params["steps_tot"], **kw_factors)],
            #     axis=None,
            # )

    params["range_agents"] = range(params["n_agents"])

    hierDict = dict()
    hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    hierDict["reg_p"] = [[0, 1, 2, 3], [3, 4, 5], [3]]
    if params["key_class"].startswith(("reg_n_p", "max_beta")):
        params["pd"] = unique_iter(
            chain.from_iterable([product(*hierDict["reg_n"]), product(*hierDict["reg_p"])])
        )
    else:  # for other simulations like "r" or "study_low_beta"
        params["pd"] = product([0, 1, 2, 3], [3], [3])
    params["pd"] = list(params["pd"])  # to prevent from exhausting generator
    return params


def get_max_betas(max_beta_dict, regType, p, n, reverse=True):
    """
    Arg
    ---
    max_beta_dict (dict): from get_params(fname="params_max_beta")
        "<regType>, <p>, <n>": [beta1, beta2, ...]
        beta1 is beta that maximizes CCS at lv1 (i.e., lv1/lv2)
    reverse (bool): whether we return the list in reverse order

    Return
    ------
    (list): since the beta decreases at the lv gets higher
        and beta is related to memory recall,
        let's assume people's memory recall gets worse as time goes on,
        then we return the list in reverse
    """
    if reverse:
        return max_beta_dict[f"{regType},{p},{n}"][::-1]
    else:
        return max_beta_dict[f"{regType},{p},{n}"]
