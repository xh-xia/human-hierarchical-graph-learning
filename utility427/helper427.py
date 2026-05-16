"""
generic helper functions

Created: Thursday, March 25, 2021, 6:25:10 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""


# region: advanced helper functions


def partial_427(func, *pargs, **pkwargs):
    """a function returning a partial function of funct (AKA partial funct)
    Note:
        for *pargs: the funct took in them first, then come *extra_args
        for **pkwargs: order doesn't matter since it operates on key:val pair.
    Arguments:
    ==========
    funct (function): function whose argments are supplied partially by pargs, pkwargs
    *pargs: partial arguments supplied, the rest (*extra_args) are to be supplied
    **pkwargs: ditto but with keywords, the rest (**extra_kwargs) are to be supplied
    """

    def wrapper(*extra_args, **extra_kwargs):
        args = list(pargs)
        args.extend(extra_args)
        kwargs = dict(pkwargs)
        kwargs.update(extra_kwargs)
        return func(*args, **kwargs)  # return what funct is supposed to return

    return wrapper  # when calling partial_427, returned thing (wrapper) is not evaluated yet


def partial_427_decorator(funct):
    """a decorator to reduce the number of arguments to whatever is left.
    Using this decorator, funct will only be supplied partial arguments,
    returning a partial function; internally this decorator uses partial_427.
    But they should not be redundant.
    Since this decorator is applied to the definition of funct,
    we only need to do it once;
    otherwise we have to call partial_427 whenever/wherever we use them.
    One may think of this decorator as "partial_427 on steroids."
    But really it doesn't do much under the hood;
    partial_427 does most of the heavy lifting.

    useful in mapping and multithreading/processing
    when only one arg (i.e., last non-kwarg arg) is expected:
    e.g., for funct = funct(x, y, z, key_x=1, key_y=2, key_z=3)
    and we want to modify it:
    @partial_427_decorator
    funct(x, y, z, key_x=1, key_y=2, key_z=3):
        whatnot
    then:
    f = funct(x, y, key_x=1, key_y=2, key_z=3) # this is funct but only taking in one arg z
    f(z) # final return
    # note: even if all arguments are passed in initially,
    # the returned object f is still a function,
    # waiting to be evaluated in order to get the return.
    """

    def wrapper(*args, **kwargs):
        return partial_427(funct, *args, **kwargs)

    return wrapper


# endregion

# region: generic helper functions


def pd_set_max_display(pd, row=15, col=4):
    """
    pandas set max display using "display.max_rows/columns" in set_option method.
    """
    pd.set_option("display.max_rows", row)
    pd.set_option("display.max_columns", col)


def set_dir427(dir_=None, add_parent_to_path=False, return_cwd=True, depth=1):
    """change working dir

    Args:
    -----
    dir_ (str):
        if present, change to that dir
        otherwise get script dir for 1-step outer frame; not script dir for helper427.py
        sys._getframe(i) (https://stackoverflow.com/questions/3711184)
            i=0: caller, which has script dir for helper427.py
            i=1: outer call (outer of caller) on the stack,
                which is where this function is called by import
                (because helper427.py should never be run directly, but imported,
                the 1-step outer frame should be in those scripts that imported helper427.py.)
    depth (int): how many outer frames away from caller
        i.e., how many calls below the top of the stack
    """
    import sys, os, inspect

    if dir_ is None:
        _cwd = os.path.dirname(os.path.abspath(inspect.getfile(sys._getframe(depth))))
        # below is the caller, which is always script dir for helper427.py
        # _cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    else:
        _cwd = dir_
    os.chdir(_cwd)  # change current working dir to where the file is
    if add_parent_to_path:  # include parent dir for .py import (non-package)
        _pwd = os.path.dirname(_cwd)  # parent dir
        sys.path.insert(0, _pwd)
    if return_cwd:
        return _cwd


def mkdir_p(dir_):
    """creates a directory. equivalent to using mkdir -p on the command line"""
    from errno import EEXIST
    from os import makedirs, path  # create new directories (i.e., folders)

    try:
        makedirs(dir_)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(dir_):  # if existed, pass
            pass
        else:
            raise


def get_params(params=None, fname="params", default_dir=True):
    """parameter management | return a dict

    Kwargs
    ------
    - default_dir (bool):
        if True, assume the parameter file is in script dir of running script + "\\input"

    Return
    ------
    it will either:
    1) create a json file named f"{fname}.json" from params if it doesn't exist
    2) read a json file named f"{fname}.json" if it exists
    either way, it will return a dictionary (params if 1), dictionary created from json if 2))

    NOTE
    ----
    params.json file should be pure json w/o comments |
    comments should be written in scripts where either
    - get_params() is called
    - or the object get_params() returns is processed
    """
    import json

    if default_dir:
        dir_ = set_dir427(depth=2) + "\\input\\"
        mkdir_p(dir_)  # create folder if it doesn't exist
        dir_ += fname + ".json"  # full absolute path of the json file
    else:
        dir_ = fname + ".json"  # relative path

    try:
        with open(dir_) as f:
            json_dict = json.load(f)  # 'tis a dict
    except FileNotFoundError:  # 1) we create json from params or just (almost) empty json
        with open(dir_, "w") as f_out:
            if params is not None:
                json.dump(params, f_out, indent=4)
            else:
                json.dump(dict(), f_out, indent=4)
        return params
    except:
        raise
    else:  # 2) json loads just fine; so we (have) read json into a dict
        return json_dict


def is_empty(x):
    from math import isnan

    if isinstance(x, int):
        return False
    elif isinstance(x, str):
        return len(x) == 0
    elif isinstance(x, float):
        return isnan(x)
    else:
        raise NotImplementedError(f"type(<x>)={type(x)} is not supported")


# endregion


def unique_iter(iter_):  # 'tis a generator function to work in tandem with itertools
    UNIQUE = set()
    for x in iter_:
        if x in UNIQUE:
            continue
        UNIQUE.add(x)
        yield x


def print427(txt, var=None):
    print(f"\n=========={txt}==========\n")
    if var is not None:
        print(var)
