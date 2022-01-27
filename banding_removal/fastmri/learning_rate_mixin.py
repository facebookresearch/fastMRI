"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import math

class LearningRateMixin(object):
    def start_of_epoch_hook(self, epoch):
        args = self.args
        momentum = args.momentum
        decay = args.decay

        if args.lr_reduction == "default":
            lr = args.lr * (0.1 ** (epoch // 75))
        elif args.lr_reduction == "none":
            lr = args.lr
        elif args.lr_reduction == "60-90":
            lr = args.lr * (0.1 ** (epoch // 60)) * (0.1 ** (epoch // 90))
        elif args.lr_reduction == "58":
            lr = args.lr * (0.1 ** (epoch // 58))
        elif args.lr_reduction == "80-120":
            lr = args.lr * (0.1 ** (epoch // 80)) * (0.1 ** (epoch // 120))
        elif args.lr_reduction == "150-225":
            lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        elif args.lr_reduction == "40-50-55":
            lr = args.lr * (0.1 ** (epoch // 40)) * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 55))
        elif args.lr_reduction == "50-55-58":
            lr = args.lr * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 55)) * (0.1 ** (epoch // 58))
        elif args.lr_reduction == "linear":
            lr = args.lr * (0.985 ** epoch)
        elif args.lr_reduction == "every20":
            lr = args.lr * (0.1 ** (epoch // 20))
        elif args.lr_reduction == "every40":
            lr = args.lr * (0.1 ** (epoch // 40))
        elif args.lr_reduction == "rampupto":
            if epoch == 0:
                lr = args.lr*0.1
            else:
                lr = args.lr
        else:
            raise Exception(f"Lr scheme not recognised: {args.lr_reduction}")

        for i, param_group in enumerate(self.optimizer.param_groups):
            if 'group_scaling' not in param_group:
                param_group['group_scaling'] = 1.0

            group_lr = lr*param_group['group_scaling']
            param_group['lr'] = group_lr
            #param_group['momentum'] = momentum
            #param_group['weight_decay'] = decay
            logging.info(f"Group {i} Learning rate: {group_lr} momentum: {momentum} decay: {decay}")

        super().start_of_epoch_hook(epoch)
