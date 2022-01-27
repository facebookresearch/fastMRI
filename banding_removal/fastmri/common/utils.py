"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import h5py
from torch.utils.data import Dataset
import gc
import torch

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


def create_submission_file(
  json_out_file, challenge, submission_url, model_name, model_description, nyu_data_only,
  participants=None, paper_url=None, code_url=None
):
    """
    Creates a JSON file for submitting to the leaderboard.
    You should first run your model on the test data, save the reconstructions, zip them up,
    and upload them to a cloud storage service (like Amazon S3).

    Args:
        json_out_file (str): Where to save the output submission file
        challenge (str): 'singlecoil' or 'multicoil' denoting the track
        submission_url (str): Publicly accessible URL to the submission files
        model_name (str): Name of your model
        model_description (str): A longer description of your model
        nyu_data_only (bool): True if you only used the NYU fastMRI data, False if you
            used external data
        participants (list[str], optional): Names of the participants
        paper_url (str, optional): Link to a publication describing the method
        code_url (str, optional): Link to the code for the model
    """

    if challenge not in {'singlecoil', 'multicoil'}:
        raise ValueError(f'Challenge should be singlecoil or multicoil, not {challenge}')

    phase_name = f'{challenge}_leaderboard'
    submission_data = dict(
        recon_zip_url=submission_url,
        model_name=model_name,
        model_description=model_description,
        nyudata_only=nyu_data_only,
        participants=participants,
        paper_url=paper_url,
        code_url=code_url
    )
    submission_data = dict(result=[{
        phase_name: submission_data
    }])

    with open(json_out_file, 'w') as json_file:
        json.dump(submission_data, json_file, indent=2)

class CallbackDataset(Dataset):
    """
        This saves memory essentially
    """
    def __init__(self, callback, start, end, increment):
        super().__init__()
        self.callback = callback
        self.start = start
        self.end = end
        self.increment = increment

    def __len__(self):
        return (self.end-self.start)//self.increment

    def __getitem__(self, i):
        return self.callback(i*self.increment+self.start)

def host_memory_usage_in_gb():
    gc.collect()
    gc.disable() # Avoids accessing gc'd objects during traversal.
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj)] # Debug
    host_tensors = [t for t in tensors if not t.is_cuda]
    total_mem_mb = 0
    visited_data = []

    for tensor in host_tensors:
        if tensor.is_sparse:
            continue
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.append(data_ptr)

        numel = tensor.storage().size()
        element_size = tensor.storage().element_size()
        mem_mb = numel*element_size /1024/1024 # 32bit=4Byte, MByte
        total_mem_mb += mem_mb

    gc.enable()
    return total_mem_mb / 1024 # in
