"""
generic helper functions

Created: Thursday, ‎March ‎25, ‎2021, ‏‎6:25:10 PM (EDT)
@author: Xiaohuan (Pixel) X.
"""


def set_dir427(dir_=None, add_parent_to_path=False, return_cwd=False):
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
    """
    import sys, os, inspect

    if dir_ is None:
        _cwd = os.path.dirname(os.path.abspath(inspect.getfile(sys._getframe(1))))
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


def unique_iter(iter_):  # 'tis a generator function to work in tandem with itertools
    UNIQUE = set()
    for x in iter_:
        if x in UNIQUE:
            continue
        UNIQUE.add(x)
        yield x
