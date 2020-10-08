import importlib
import inspect
import pdb
import logging

def load(name):
    logging.info(f"Loading transformer {name}")
    # Split name into file and method name?
    parts = name.rsplit(".", 1)
    module_name = parts[0]
    if len(parts) == 1:
        class_name = "default"
    else:
        class_name = parts[1]
    try:
        mdl = importlib.import_module("." + module_name, "fastmri.transforms")
        classobj = getattr(mdl, class_name)
    except AttributeError as e:
        raise Exception(f"{name} method in specified transformer module doesn't exist")
    except ModuleNotFoundError as e:
        raise Exception(f"{module_name} transformer module file not found")
    return classobj
