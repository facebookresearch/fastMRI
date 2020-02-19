from .base_trainer import BaseTrainer
from .checkpointing_mixin import CheckpointingMixin
from .distributed_mixin import DistributedMixin
from .learning_rate_mixin import LearningRateMixin
from .logging_mixin import LoggingMixin
from .training_loop_mixin import TrainingLoopMixin
from .transform_mixin import TransformMixin
from .visualization_mixin import VisualizationMixin
from .parameter_group_mixin import ParameterGroupMixin
from .ssim_loss_mixin import SSIMLossMixin
from .orientation_adversary.adversary_mixin import AdversaryMixin

class Trainer(object):
    def __new__(cls, args):
        bases = []
        if args.orientation_adversary:
            bases.append(AdversaryMixin)
        if args.is_distributed:
            bases.append(DistributedMixin)
        if args.ssim_loss:
            bases.append(SSIMLossMixin)
        if args.parameter_groups:
            bases.append(ParameterGroupMixin)

        bases += [VisualizationMixin, LoggingMixin, LearningRateMixin, CheckpointingMixin,
                 TransformMixin, TrainingLoopMixin, BaseTrainer]

        trainer = super().__new__(cls)
        cls = trainer.__class__
        trainer.__class__ = cls.__class__(cls.__name__, (cls, *bases), {})
        return trainer
