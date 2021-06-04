"""
Helper functions 427

Created: Tuesday, ‎May ‎4, ‎2021, ‏‎11:18:41 AM (EDT)
@author: Pixel
"""


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