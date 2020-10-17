"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("../../")

import fastmri
import fastmri.data.transforms as T
from experimental.varnet.varnet_module import DataTransform
from fastmri.data import SliceDataset
from fastmri.models import VarNet


def run_model(masked_kspace, mask, varnet, fname, device):
    masked_kspace = masked_kspace.to(device)
    varnet = varnet.to(device)
    output = varnet(masked_kspace, mask.to(device)).to(torch.device("cpu"))

    return output


def run_inference(checkpoint, data_path, output_path):
    varnet = VarNet()
    load_state_dict = torch.load(checkpoint)["state_dict"]
    state_dict = {}
    for k, v in load_state_dict.items():
        if "varnet" in k:
            state_dict[k[len("varnet.") :]] = v

    varnet.load_state_dict(state_dict)
    varnet = varnet.eval()

    data_transform = DataTransform()

    dataset = SliceDataset(
        root=data_path, transform=data_transform, challenge="multicoil",
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    start_time = time.perf_counter()
    outputs = defaultdict(list)

    for batch in tqdm(dataloader, desc="Running inference..."):
        masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        crop_size = crop_size[0]  # always have a batch size of 1 for varnet
        fname = fname[0]  # always have batch size of 1 for varnet

        with torch.no_grad():
            try:
                device = torch.device("cuda")

                output = run_model(masked_kspace, mask, varnet, fname, device)
            except RuntimeError:
                print("running on cpu")
                device = torch.device("cpu")

                output = run_model(masked_kspace, mask, varnet, fname, device)

            output = T.center_crop(output, crop_size)[0]

        outputs[fname].append((slice_num, output))

    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--data-path", type=pathlib.Path, required=True, help="Path to subsampled data",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(args.checkpoint, args.data_path, args.output_path)
