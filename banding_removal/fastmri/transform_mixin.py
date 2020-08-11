import pdb
from . import transforms
from .common.subsample import mask_factory
from .transforms.orientation import Orientation

class TransformMixin(object):
    def transform_setup(self, args):
        train_mask = mask_factory(args.mask_type, args.train_num_low_frequencies, args.train_accelerations)
        dev_mask = mask_factory(args.mask_type, args.num_low_frequencies, args.accelerations)

        Transform = transforms.load(args.data_transform)

        self.dev_transform = Transform(args, dev_mask, partition='val', use_seed=True)
        self.train_transform = Transform(args, train_mask, partition='train')

        if args.orientation_augmentation:
            self.train_transform = Orientation(after=self.train_transform, args=args)

        super().transform_setup(args)
