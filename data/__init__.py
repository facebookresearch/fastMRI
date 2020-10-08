"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings

dep_str = """Importing from data has been deprecated and will be removed in a future update. Please import from fastmri instead."""
warnings.warn(dep_str)
