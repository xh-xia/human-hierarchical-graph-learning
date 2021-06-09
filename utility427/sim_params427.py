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


def make_sim_params(params):
    """set up parameters for simulations

    Arg - params (dict)
    -------------------
    NOTE: params should be read from params.json file, which should be pure json w/o comments
    - key_classes (list): list of key_class (currently only uses first entry)
        - key_class (str): type of simulation run; encoding regType, p, n parameters
            e.g., "reg_n_p", "r"
    - n_agents (int): num of agents per param (including beta)
    - n_beta_constcase (int): num of beta in constant beta case
    - SEED (int): seed for current simulation params
    - steps_tot (int): total number of steps for random walk
    - sample_period (int): sample sim results every <sample_period> steps
    - var_beta (bool): whether we extend codenames (variable beta case) in "beta_arr"
    - ub (int): upperbound for variable beta (shows up in get_factors())

    Intermediary
    ------------
    - params["pd"] (iter): reg, n, p tuple iterator
    - params["int_max"] (int32): some constant, in this case 2147483647 for int (int32 really)
    - params["beta_arr"] (np.arr): e.g., [0.01, 0.1, 1,| -1, -2, -3]
                            (constant case)|(variable case)
        1D arr of beta values (constant beta case)
        also incorporates variable beta case (mono-increase step function w/ certain <sf_width>)
        sweeping from beta_min (min(constant case)) to beta_max (max(constant case))
            val (<0; comes after constant case) in the arr is only a codename:
            -x (x is positive int):
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
    - params["range_agents"] (range object): range(params["n_agents"])
    """
    params["int_max"] = np.iinfo(int).max

    params["beta_arr"] = 10 ** np.linspace(-3, 1, params["n_beta_constcase"], endpoint=True)
    if params["var_beta"]:
        params["beta_arr"] = np.concatenate(
            [params["beta_arr"], get_factors(params["steps_tot"], neg=True, ub=params["ub"])],
            axis=None,
        )
    params["range_agents"] = range(params["n_agents"])

    hierDict = dict()
    hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    hierDict["reg_p"] = [[0, 1, 2, 3], [3, 4, 5], [3]]
    if params["key_classes"][0] == "reg_n_p":
        params["pd"] = unique_iter(
            chain.from_iterable([product(*hierDict["reg_n"]), product(*hierDict["reg_p"])])
        )
    elif params["key_classes"][0] == "r":
        params["pd"] = product([0, 1, 2, 3], [3], [3])
    return params
