import pathlib
import sys
from getpass import getuser
sys.path.append(sys.path[0] + "/../..")

from fastmri import run, spawn_dist

config = {
    'run_name': pathlib.Path(__file__).stem,
    'data_path': '/datasets01_101/fastMRI/112718',
    'trainer_class': 'fastmri.var_net.var_net_trainer.VarNetTrainer',
    'architecture': 'var_net.var_net',
    'data_transform': 'kspace.KSpaceDataTransform',

    'method_str': "12[SoftDC(cunet(18,2)),]IFT(),RSS()",
    'sens_method_str': 'MaskCenter(),IFT(),Fm2Batch(cunet(8,2)),dRSS()',
    'mask_type': 'magic', # Key change

    'ssim_loss': True,
    'ssim_l1_coefficient': 0.01,
    'epochs': 100,

    'batch_size': 1,
    'method': 'adam',
    'lr': 0.0003,

    'workers': 8,

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
    'sync_params': False,

    'log_interval': 10,
    'visual_first_epoch': False,

    'orientation_augmentation': True,
    'orientation_adversary': False,
}

if __name__ == "__main__":
    spawn_dist.run(config)
