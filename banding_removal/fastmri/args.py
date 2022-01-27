"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import sys
import pickle
from pathlib import Path

class Args(argparse.ArgumentParser):
    """
        Contains global default arguments for experimentation.
        Override by passing in a dict on initialization.
    """

    def __init__(self, **overrides):
        super().__init__()
        self.overrides = overrides

        self.add_argument('--run_name', default="run", type=str)
        self.add_argument('--seed', default=42, type=int)
        self.add_argument('--debug', default=False, type=bool,
            help="Additional debug logging/checks. Very slow.")
        self.add_argument('--strace', default=False, type=bool,
            help="Monitor stdout using strace, for detailed logging of distributed runs")
        self.add_argument("--trainer_class", default="fastmri.trainer.Trainer", type=str)
        self.add_argument("--architecture", default="public_unet.unet", type=str)
        self.add_argument('--gan', dest='gan', action='store_true')
        self.add_argument('--eval', dest='eval', action='store_true')
        self.add_argument('--copy_data_to_scratch', dest='copy_data_to_scratch', action='store_true')
        self.add_argument('--override_with_runinfo_args', type=Path, default=None,
                help='If specified, override the args with the ones given in the provided runinfo file')

        self.add_argument('--resize_type', default="crop", choices=["none", "crop", "pad", 'singlebatch'],
                help='How to scale the images to the given resolution')
        self.add_argument("--data_transform", default="rss.RSSDataTransform", type=str)
        self.add_argument('--resolution_width', default=320, type=int, help='Resolution width of images')
        self.add_argument('--resolution_height', default=320, type=int, help='Resolution height of images')

        self.add_argument('--grappa_input', type=bool, default=False, help='Should you use Grappa for input')
        self.add_argument('--grappa_input_path', type=Path, default=None, help='Path to grappa kernels')
        self.add_argument('--grappa_path', type=Path, default=None, help='Path to grappa kernels')
        self.add_argument('--grappa_target', type=bool, default=False, help='Should you use 2x grappa for the ground truth')
        self.add_argument('--grappa_target_path', type=Path, default=None, help='Path to grappa kernels')
        self.add_argument('--apply_grappa', type=bool, default=True, help='Should you use Grappa in the kspace model')

        self.add_argument('--display_ifft', dest="display_ifft", action='store_true')

        self.add_argument('--challenge', choices=['singlecoil', 'multicoil'], default="multicoil",
            help='Which challenge')

        self.add_argument('--magnet', type=int, default=None)
        self.add_argument('--data_path', type=Path, default="/datasets01_101/fastMRI/112718",
            help='Path to challenge dataset')
        self.add_argument('--sample_rate', type=float, default=1.,
            help='Fraction of total volumes to include')
        self.add_argument('--calculate_offsets_directly', type=bool, default=False,
            help="Ignore acq start/end info in metadata and just calculate directly")
        self.add_argument('--min_kspace_width', type=int, default=None)
        self.add_argument('--max_kspace_width', type=int, default=None)
        self.add_argument('--min_kspace_height', type=int, default=None)
        self.add_argument('--max_kspace_height', type=int, default=None)
        self.add_argument('--min_target_width', type=int, default=None)
        self.add_argument('--min_target_height', type=int, default=None)
        self.add_argument('--max_target_width', type=int, default=None)
        self.add_argument('--max_target_height', type=int, default=None)
        self.add_argument('--start_slice', type=int, default=None)
        self.add_argument('--end_slice', type=int, default=None)
        self.add_argument('--only_square_targets', type=bool, default=False)
        self.add_argument('--filter_acceleration', type=int, default=None, help="Filter input data that has accelerated by the given amount")
        self.add_argument('--max_num_coils', type=int, default=None)
        self.add_argument('--min_num_coils', type=int, default=None)
        self.add_argument('--scale_inputs', type=bool, default=True)
        self.add_argument('--scale_type', type=str, default='mean')
        self.add_argument('--acquisition_types', nargs='*', default=None)
        self.add_argument('--acquisition_systems', nargs='*', default=None, help="Choices: ['Avanto', 'TrioTim', 'Skyra', 'Aera', 'Biograph_mMR', 'Prisma_fit']")
        self.add_argument('--coil_compress_coils', type=int, default=None)

        # var net args
        self.add_argument('--method-str', help='The main model is build from this string in var_net.var_net')
        self.add_argument('--sens-method-str', help='The sensitivty maps model is build from this string in var_net.var_net')
        self.add_argument('--norm', choices=('layer', 'instance', 'group'), default='group', help='Normalization of the unets in var_net.var_net')
        self.add_argument('--norm-type', choices=('layer', 'instance', 'group'), default='group')
        self.add_argument('--norm-mean', type=int, default=1)
        self.add_argument('--norm-std', type=int, default=1)
        self.add_argument('--kernel-size', type=int, default=3, help='Kernel size in convolutions in var_net.var_net')
        self.add_argument('--sqrt-eps', type=float, default=0, help='Number to add before calling .sqrt() in var_net.var_net')
        self.add_argument('--var_net_model', type=str, default='unet', help='The UNet model used in var net')
        self.add_argument('--compute_sensitivities', type=bool, default=False)

        # Mask parameters
        self.add_argument('--accelerations', nargs='+', default=[4], type=int,
            help='Ratio of k-space columns to be sampled. If multiple values are '
                 'provided, then one of those is chosen uniformly at random for '
                 'each volume.')
        self.add_argument('--num_low_frequencies', nargs='+', default=[28], type=int,
            help='Number of low-frequency k-space columns to be sampled. Should '
                 'have the same length as accelerations')
        self.add_argument('--train_accelerations', nargs='+', default=[4], type=int,
                help='Equivalent to --accelerations but for train set')
        self.add_argument('--train_num_low_frequencies', nargs='+', default=[28], type=int,
                help='Equivalent to --num_low_frequencies but for train st')
        self.add_argument('--mask_type', default="equispaced", choices=["equispaced", "random", "random_fraction", "magic"],
            help='The strategy used to mask k-space inputs')

        #### Data augmentations
        self.add_argument('--transforms_on_gpu', default=False, type=bool,
            help="Do data transforms on the gpu when possible")
        self.add_argument('--padding_augmentation', default=0, type=int,
            help="Number of levels of padding augmentation to randomly sample from")
        self.add_argument('--rotation_augmentation', default=False, type=bool)
        self.add_argument('--elastic_augmentation', default=False, type=bool)
        self.add_argument('--orientation_augmentation', default=False, type=bool)
        self.add_argument('--orientation_augmentation_dev', default=False, type=bool)
        self.add_argument('--debug_phase_direction', default=False, type=bool, help="For debugging only")
        self.add_argument('--add_gibbs_artifacts_augmentation', default=False, type=bool,
            help="Add false Gibbs artifacts to the image")

        self.add_argument('--resize_min_width', default=None, type=int,
            help="Input is resized in image space to if less wide than this")
        self.add_argument('--resize_max_width', default=None, type=int,
            help="Input is resized in image space to if wider than this")

        self.add_argument('--batch_size', default=20, type=int, help="Per-gpu batch-size")
        self.add_argument('--eval_batch_size', default=-1, type=int,
            help="Larger batches can be used during eval as less memory is needed")

        self.add_argument('--workers', default=8, type=int, help="Data loader worker count per GPU")
        self.add_argument('--pin_memory', default=False, type=bool,
            help="Pin tensors in dataloader. Can cause issues with distributed training")
        self.add_argument('--is_distributed', default=False, type=bool, help="Distributed training flag (set automatically)")
        self.add_argument('--use_barriers', default=True, type=bool,
            help="During distributed training, keep processes in sync using barriers")
        self.add_argument('--sync_params', default=False, type=bool,
            help="Sync paramters between machines each epoch to reduce drift, and diagnose initialization bugs")
        self.add_argument('--rank', default=0, type=int,
            help="Distributed rank (0 for single). Set by environment variable automatically")
        self.add_argument('--world_size', default=1, type=int,
            help="Distributed world_size (1 for single). Set by environment variable automatically")
        self.add_argument('--apex_distributed', default=False, type=bool, help="Use Apex for distributed training")

        self.add_argument('--apex', default=False, type=bool, help="NVIDIA Apex half-prec training")
        self.add_argument('--apex_amp', default=False, type=bool)
        self.add_argument('--apex_loss_scale', default=1e4, type=float,
            help="Prevent underflow by scaling the loss internally up by this amount")
        self.add_argument('--nan_detection', default=False, type=bool,
            help="Use pytorch's NaN detection mode which is slower but extremely useful for debugging")

        self.add_argument('--log_interval', default=10, type=int)
        self.add_argument('--save_info', default=True, type=bool)
        self.add_argument('--save_model', default=True, type=bool)
        self.add_argument('--display_count', default=16, type=int,
            help="How many images to save out every epoch for display")
        self.add_argument('--visual_first_epoch', default=True, type=bool,
            help="save image grid visual of untrained model before training begins")


        self.add_argument('--method', default='rmsprop', type=str)
        self.add_argument('--lr', default=0.001, type=float)
        self.add_argument('--momentum', default=0.9, type=float)
        self.add_argument('--beta2', default=None, type=float)
        self.add_argument('--adam_eps', default=1e-8, type=float)
        self.add_argument('--decay', default=0.0, type=float)
        self.add_argument('--lr_reduction', default="every40", type=str)
        self.add_argument('--parameter_groups', default=False, type=bool,
            help="Split scalar and vector parameters into separate parameter groups")
        self.add_argument('--bias_lr_scale', default=0.1, type=float,
            help="Scale the learning rate of all scalar/vector model parameters by this amount if --parameter_groups is set")
        self.add_argument('--ramp_lr_by', default=1, type=int,
            help="Start learning rate x times smaller and ramp up") # Not implemented yet
        self.add_argument('--epochs', default=50, type=int)
        self.add_argument('--eval_at_start', default=False, type=bool,
            help="Perform DEV set evaluation at the beginning of training")
        self.add_argument('--debug_epoch', default=False, type=bool,
            help="Only process one batch each epoch for debugging purposes")
        self.add_argument('--debug_epoch_stats', default=False, type=bool,
            help="Only process one batch each stats (eval) epoch for debugging purposes")
        self.add_argument('--break_early', default=None, type=float,
            help="Percentage to break each epoch at")
        self.add_argument('--debug_memory', default=False, type=bool, help='Output memory diagnostics then quit')
        self.add_argument('--channels', default=2, type=int)

        # UNET Settings
        self.add_argument('--num_chans', type=int, default=128, help='Number of U-Net channels')
        self.add_argument('--res_chans', type=int, default=128, help='Number of U-Net channels')
        self.add_argument('--num_pools', type=int, default=4, help='Number of U-Net pooling layers')
        self.add_argument('--drop_prob', type=float, default=0.0, help='Dropout probability')
        self.add_argument('--num_models', type=int, default=4, help='Number of models for cascaded models')
        self.add_argument('--dilation', type=int, default=1, help='Conv dilation')
        self.add_argument('--num_layers', type=int, default=4, help='Num of conv layers')
        self.add_argument('--groups', type=int, default=1, help='Num of connections between inputs and outputs')

        # Initialization
        self.add_argument('--smart_initialization', default=False, type=bool)
        self.add_argument('--initialization', default="fan_out", type=str)
        self.add_argument('--dropout',  default=False, type=bool)

        self.add_argument('--exp_dir', type=Path, default=Path.cwd() / "logs" / "run",
            help='Path where model and results should be saved')

        # Dicom
        self.add_argument('--dicom', default=False, type=bool,
            help='Use dicom dataset rather than challenge')
        self.add_argument('--dicom_root', type=Path, default="/checkpoint/jzb/data/mmap",
            help='Path to dicom dataset')
        self.add_argument('--dicom_normalization', default="volume", choices=["volume", "instance"],
            help='Normalize dicom images by volume or instance mean and std.')
        self.add_argument('--filter_dicom_scan_type', default=False, type=bool,
            help='Filter dicom dataset type by the "cor" scan type. These images are closer '
                'to the challenge dataset.')

        # Perecptual loss
        self.add_argument('--perceptual', type=bool, default=False,
                          help='Use perceptual loss while training')
        self.add_argument('--perceptual_loss_normalize', type=bool, default=False,
                          help="Prenormalize inputs to the perceptual loss to be mean 0 variance 1")
        self.add_argument('--perceptual_loss_architecture', type=str,
                          default='vgg_bw_perceptual.vgg19bw_features',
                          help='Base architecture of the perceptual loss net')
        self.add_argument('--perceptual_loss_kwargs', type=dict, default={},
                          help='kwargs passed to the perceptual loss architecture')
        self.add_argument('--perceptual_loss_checkpoint', type=Path,
                          default='/checkpoint/mikerabbat/fast_mri/perceptual/pretrained_models/vgg19bw/checkpoint.pt')
        self.add_argument('--perceptual_loss_cutoff_layer', type=str,
                          default='relu2_2',
                          help='Layer of the base architecture to use as '
                               'features for perceptual loss')
        self.add_argument('--perceptual_l1_coefficient', type=float,
                          default=0.1,
                          help='Weight multiplying L1 loss added to perceptual')

        self.add_argument('--ssim_loss', type=bool, default=False,
                          help='Use SSIM loss while training')
        self.add_argument('--ssim_l1_coefficient', type=float, default=0.0,
                          help='Weight multiplying L1 loss added to SSIM loss')

        self.add_argument('--q_loss', type=str, default=None,
                          help='Select a q weighted loss')
        self.add_argument('--loss_denominator_power', default=1.0, type=float,
            help="q weight for the q_loss, changing relative weighting of low vs high contrast regions")
        self.add_argument('--loss_kernel', type=float, default=1.5,
                          help='Gaussian kernel size in pixel stds used for loss calculations')

        self.add_argument('--gradient_loss', type=bool, default=False,
            help="Penalize differences in the average magnitude of the gradient")

        # DicomWriter
        self.add_argument('--model_checkpoint', default=None, type=Path,
                          help='Path to the model checkpoint to use for reconstruction'
                            '(Used by DicomWriter)')
        self.add_argument('--dicom_save_dir', default=None, type=Path,
                          help='Path to the directory where dicoms should be saved')
        self.add_argument('--save_ground_truth', default=False, type=bool,
                          help='Whether to save DICOMs of ground truth images (default=False)')
        self.add_argument('--reconstruct_all', default=False, type=bool,
                          help='Save DICOMs of reconstructions for every test case.'
                            'Default is to only save the six common test cases'
                            'used in the image review sessions.')
        self.add_argument('--noise_levels', nargs='*', default=[], type=float,
                          help='List of noise levels used when producing DICOMS')
        self.add_argument('--series_number', default=0, type=int,
                          help='Initial series number (useful when presenting '
                            'multiple methods side-by-side)')
        self.add_argument('--series_description', default='', type=str,
                          help='Descriptive name for this series')

        ### Cluster
        self.add_argument('--checkpoint_type', default="none", choices=["resume", "restart", "none"],
            help='Resume (keeping all runinfo) or restart (keeping only model weights) from a previous '
                'model checkpoint. "--checkpoint" should be set with this')

        self.add_argument('--checkpoint', type=str,
            help='Path to an existing checkpoint. Used along with "--resume"')
        self.add_argument('--auto_requeue', default=False, type=bool,
            help='If job is killed by slurm, reschedule it')

        ##############################
        ### Direct zeronet parameters
        self.add_argument('--first_layer_planes', default=128, type=int)
        self.add_argument('--cascades', default=3, type=int)
        self.add_argument('--blocks_middle', default=4, type=int)
        self.add_argument('--blocks_inner_middle', default=2, type=int)
        self.add_argument('--blocks_inner', default=2, type=int)
        self.add_argument('--blocks_outer', default=1, type=int)
        self.add_argument('--use_fixed_conv', default=False, type=bool)
        self.add_argument('--bottle_neck_factor', default=1, type=int, help="resnet block expansion factor")
        self.add_argument('--groupnorm', default=True, type=bool)
        self.add_argument('--use_fixed_conv_block', default=False, type=bool)
        self.add_argument('--block_type', default='bottleneck', type=str)
        self.add_argument('--tall_convs', default=False, type=bool)

        self.add_argument('--autoregressive_sample_rate', default=0.01, type=float)
        self.add_argument('--ar_channels', default=256, type=int)
        self.add_argument('--box_sample', default=True, type=bool,
            help="Do AR sampling simultanously for each box, with the boxes tiling the image")
        self.add_argument('--boxes_per_dim', default=10, type=int,
            help="Tile image with the square of this many boxes. Must evenly divide height/width")

        #### Autocalibration
        self.add_argument('--autocal_subsample', default=False, type=bool,
            help="Train autocalibration model on subsampled k-space or not")
        self.add_argument('--autocal_soft_normalization', default=False, type=bool)

        ## Direct reconstruction
        self.add_argument('--whiten_coils', default=True, type=bool,
            help="Pre-whiten all coils for the direct transforms")
        self.add_argument('--sensitivity_target', default=True, type=bool,
            help="use SENSE-REDUCE based target instead of RSS for the direct transforms")
        self.add_argument('--imagespace_projection', default=False, type=bool,
            help="Use imagespace projection instead of fourier based projection")
        self.add_argument('--rss_output', default=False, type=bool,
            help="Use RSS instead of SENSE based output on the last layer of the cascade")
        self.add_argument('--gradient_checkpointing', default=False, type=bool,
            help="Greatly reduce memory using checkpointing. Requires support by the model used.")

        ### Architecture search
        self.add_argument('--arch_seed', default=None, type=int,
            help="Optionanlly fix architecture every iteration with the archsearch model, for debugging")
        self.add_argument('--arch_eval_size', default=10, type=int,
            help="Size of subset of the training data used to choose best architecture (per gpu)")
        self.add_argument('--arch_eval_nseeds', default=1000, type=int,
            help="Number of seeds to consider during post-run search for best architecture")

        #### Orientation adversary
        self.add_argument('--orientation_save_images', default=None, type=str,
            help="For one-off use, save out pngs for debugging")
        self.add_argument('--orientation_save_images_from', default=None, type=int,
            help="skip this number of initial batches")
        self.add_argument('--orientation_adversary', default=False, type=bool,
            help="Use the orientation detection adversary")
        self.add_argument('--number_of_adversaries', default=1, type=int,
            help="Potentially use an ensemble of multiple adversaries")
        self.add_argument('--adversary_lr_scale', default=1.0, type=float,
            help="The learning rate for the adversary is this multiple of the main learning rate")
        self.add_argument('--adversary_epoch_from', default=0, type=int,
            help="Start training adversary at a certain epoch")
        self.add_argument('--warmup_adversary_from', default=0, type=int,
            help="Start training adversary from this epoch, potentially from an earlier epoch than "
            "when it's used to regulate the predictor via adversary_epoch_from")
        self.add_argument('--adversary_strength', default=1.0, type=float,
            help='strength of adversary_mixin adv training in loss')
        self.add_argument('--reg_param', default=0.0, type=float,
            help="Regularize the adversary's gradient norm")
        self.add_argument('--adversary_weight_decay', default=0.0, type=float,
            help="Apply this weight decay to the adversary (decay param effects predictor only)")
        self.add_argument('--adv_target_uncertain', default=False, type=bool,
            help="Train predictor to encourage 0.5 prop output from adv instead of 0/1")
        self.add_argument('--adversary_model', default="unpooled_resnet50", type=str)
        self.add_argument('--dont_learn_predictor', default=False, type=bool,
            help="Only learn adversary for debugging purposes")

        # multi-slice paarams
        self.add_argument('--before_slices', default=0, type=int,
            help="number of slices to grab for prediction before the slice being predicted")
        self.add_argument('--after_slices', default=0, type=int,
            help="number of slices to grab for prediction after the slice being predicted")

        # Override defaults with passed overrides
        self.set_defaults(**overrides)

        ## Some run specific context that we want globally accessible and saved out
        ## at the end of the run.
        self.set_defaults(
            main_pid = os.getpid(),
            cwd = os.getcwd()
        )

    def parse_args(self, args=None, namespace=None):
        """
            As well as the usual command line argument parsing, this also accepts a
            single argument specifying a pickle file, which arguments are read from.

            It also overrides defaults if the override_with_runinfo_args argument is provided.
        """
        args_list = sys.argv
        if len(sys.argv) == 2 and "--" not in sys.argv[1] and ".pkl" in sys.argv[1]:
            run_config_file = sys.argv[1]
            run_config = pickle.load(open(run_config_file, 'rb'))
            if not isinstance(run_config, list):
                if int(os.environ.get('RANK', 0)) == 0:
                    print(f"Found single argument config file: {run_config_file}. Overriding defaults")
                if not isinstance(run_config, dict):
                    run_config = vars(run_config)
                self.set_defaults(**run_config)
            parsed_args = super().parse_args([])
        else:
            parsed_args, _ = super().parse_known_args(args=args, namespace=namespace)

        if parsed_args.override_with_runinfo_args is not None:
            run_config = pickle.load(open(parsed_args.override_with_runinfo_args, 'rb'))
            run_config_args = vars(run_config['args'])

            # We need to override the defaults twice so that the user defined arguments
            # always have the highest priority, followed by the runinfo args and then the defaults.
            self.set_defaults(**run_config_args)
            self.set_defaults(**self.overrides)
            parsed_args, _ = super().parse_known_args(args=args, namespace=namespace)

        return parsed_args
