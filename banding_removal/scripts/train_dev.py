"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from getpass import getuser
sys.path.append(sys.path[0] + "/../..")

from fastmri import run, spawn_dist

config = {
    'run_name': pathlib.Path(__file__).stem,
    'data_path': '/datasets01/fastMRI/112718', #SET 
    'trainer_class': 'fastmri.var_net.var_net_trainer.VarNetTrainer',
    'architecture': 'var_net.var_net',
    'data_transform': 'kspace.KSpaceDataTransform',

    'method_str': "12[SoftDC(cunet(18,2)),]IFT(),RSS()",
    'sens_method_str': 'MaskCenter(),IFT(),Fm2Batch(cunet(8,2)),dRSS()',
    'mask_type': 'magic',

    'ssim_loss': True,

    'batch_size': 1, # Larger batches are not supported
    'method': 'adam',
    'lr': 0.0001, # Adjust based on number of gpus

    'calculate_offsets_directly': True,

    'workers': 0, # Process data in main process for debugging purposes

    'filter_acceleration': 1,
    'min_kspace_width': None,
    'max_kspace_width': 740,
    'min_kspace_height': None,
    'max_kspace_height': None,
    'acquisition_type': None,
    'system_type': None,
    'min_target_width': None,
    'min_target_height': None,
    'only_square_targets': True,
    'scale_inputs': False,
    'num_coils': 15,
    'magnet': 1.5,

    'accelerations': [4],
    'num_low_frequencies': [16],
    'train_accelerations': [4],
    'train_num_low_frequencies': [16],

    'initialization': 'none',

    'log_interval': 10,
    'visual_first_epoch': False,
    'display_count': 8,

    # After pretraining, link to pretrained model file here
    'checkpoint_type': "restart",
    'checkpoint': f'{pathlib.Path.home()}/pretrained_lowt_4x.mdl',

    'orientation_augmentation': True,
    'orientation_adversary': True,

    'adversary_model': 'shallow',
    'adversary_epoch_from': 0,
    'warmup_adversary_from': 0,
    'adversary_strength': 1,

    'reg_param': 0.1,
    'adversary_weight_decay': 0.0,
    'ssim_l1_coefficient': 0.01,
    'number_of_adversaries': 1,

    'evaluate': False,
    'short_epochs': True, # 20% epochs to avoid memory issues
    'epochs': 1000,
}

if __name__ == "__main__":
    run.run(config) # Single GPU training for debugging only
