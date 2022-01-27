"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib
import inspect
import pdb
import logging

def load(name, args):
    logging.info(f"Loading architecture {name}")
    # Split name into file and method name?
    parts = name.rsplit(".", 1)
    module_name = parts[0]
    if len(parts) == 1:
        method_name = "default"
    else:
        method_name = parts[1]
    try:
        mdl = importlib.import_module("." + module_name, "fastmri.model")
        methodobj = getattr(mdl, method_name)
    except AttributeError as e:
        raise Exception(f"{name} method in specified architecture module doesn't exist")
    except ModuleNotFoundError as e:
        raise Exception(f"{module_name} architecture module file not found")
    return methodobj(args)
