import logging
import math

class ParameterGroupMixin(object):
    """
        We separate any scalar parameters into a separate group,
        and give then a 10x smaller learning rate.
    """
    def parameter_groups_setup(self, args):
        main_group = []
        bias_group = []
        scalar_group = []

        for p in self.model.parameters():
            if p.dim() == 0:
                scalar_group.append(p)
            elif p.dim() == 1:
                bias_group.append(p)
                #print(p.shape)
            else:
                main_group.append(p)


        self.parameter_groups = [
                    {'params': main_group},
                    {'params': bias_group, 'group_scaling': args.bias_lr_scale},
                    {'params': scalar_group, 'group_scaling': args.bias_lr_scale}
        ]
        logging.info(f"Parameter groups | Main: {len(main_group)} Scalar: {len(scalar_group)} Bias: {len(bias_group)}")
