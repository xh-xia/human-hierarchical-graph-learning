"""
Helper functions 427

Created: Thursday, ‎March ‎25, ‎2021, ‏‎6:25:10 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""
import numpy as np
from itertools import product, chain, repeat
import concurrent.futures  # multiprocessing

def set_dir427(add_parent_to_path=False, return_cwd=False):
    import os, inspect

    _cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    os.chdir(_cwd)  # change current working dir to where the file is
    if add_parent_to_path:  # include parent dir for .py import (non-package)
        import sys

        _pwd = os.path.dirname(_cwd)  # parent dir
        sys.path.insert(0, _pwd)
    if return_cwd:
        return _cwd


def mkdir_p(path_):
    """creates a directory. equivalent to using mkdir -p on the command line"""
    from errno import EEXIST
    from os import makedirs, path  # create new directories (i.e., folders)

    try:
        makedirs(path_)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(path_):  # if existed, pass
            pass
        else:
            raise

def unique_iter(iter):  # 'tis a generator function to work in tandem with itertools
    UNIQUE = set()
    for x in iter:
        if x in UNIQUE:
            continue
        UNIQUE.add(x)
        yield x


def get_params(n_agents=10, key_class="reg_n_p"):
    """set up parameters for simulations
    n_agents: num of agents per param (including beta)
    int_max: 2147483647 for int (int32 really)
    """
    params = {
        "key_classes": [key_class],
        "n_agents": n_agents,
        "SEED": 427,
        "steps_tot": 3000,
        "sample_period": 1500,
        "int_max": np.iinfo(int).max,
    }

    params["beta_arr"] = 10 ** np.linspace(-3, 1, 13, endpoint=True)
    params["range_agents"] = range(params["n_agents"])

    hierDict = dict()
    hierDict["reg_n"] = [[0, 1, 2, 3], [3], [3, 4, 5]]
    hierDict["reg_p"] = [[0, 1, 2, 3], [3, 4, 5], [3]]
    if key_class == "reg_n_p":
        params["pd"] = unique_iter(
            chain.from_iterable([product(*hierDict["reg_n"]), product(*hierDict["reg_p"])])
        )
    elif key_class == "r":
        params["pd"] = product([0, 1, 2, 3], [3], [3])
    return params
