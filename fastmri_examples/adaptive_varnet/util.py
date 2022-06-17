import pathlib
import shutil
import tempfile
from typing import Tuple


def copy_file(src_dest_tuple: Tuple[pathlib.Path, pathlib.Path]):
    src = src_dest_tuple[0]
    dest = src_dest_tuple[1]
    if dest.is_file():
        print(f"Found {dest}, nothing to copy.")
        return
    print(f"Copy from {src} to {dest}")
    temp, temp_fname = tempfile.mkstemp(dir=dest.parent)
    shutil.copy(src, temp_fname)
    shutil.move(temp_fname, dest)
