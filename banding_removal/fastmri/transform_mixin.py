from . import transforms
from .common.subsample import mask_factory
import pdb

class TransformMixin(object):
    def transform_setup(self, args):
        train_mask = mask_factory(args.mask_type, args.train_num_low_frequencies, args.train_accelerations)
        dev_mask = mask_factory(args.mask_type, args.num_low_frequencies, args.accelerations)

        Transform = transforms.load(args.data_transform)

        self.dev_transform = Transform(args, dev_mask, partition='val', use_seed=True)
        self.train_transform = Transform(args, train_mask, partition='train')

        super().transform_setup(args)
