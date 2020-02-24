
import torch
from torch.nn import functional as F
from torch import nn
from torch import autograd

import pdb
import logging
from fastmri.data import transforms
from fastmri import model
from fastmri.model.classifiers.torchvision_resnet import Resnet50
from fastmri.model.classifiers.unpooled_resnet import UnpooledResnet50
from fastmri.model.classifiers.resnet_r1 import Discriminator
from fastmri.model.classifiers.resnet_r1_wide import WideDiscriminator
from fastmri.model.classifiers.resnet_r1_simple import SimpleDiscriminator

class AdversaryModel(nn.Module):
    """
        By storing the adversary in the same model object
        we avoid a lot of extra boilerplate code.
    """
    def __init__(self, prediction_model, adversary_model):
        super().__init__()
        self.prediction_model = prediction_model
        self.adversary_model = adversary_model
    def forward(self, *args, **kwargs):
        """
            Evaluating the adversary and prediction models both need
            to use the forward method to work with distributed learning
        """
        if 'adversary' in kwargs and kwargs['adversary']:
            return self.adversary_model(*args) # Doesn't take kwargs
        else:
            return self.prediction_model(*args, **kwargs)
        
class AdversaryEnsemble(nn.Module):
    """
        Multiple adversaries
    """
    def __init__(self, nadvs, Adversary, **kwargs):
        super().__init__()
        self.adversaries = nn.ModuleList()
        for i in range(nadvs):
            self.adversaries.append(Adversary(**kwargs))
        
    def forward(self, *args):
        # Apply to each and concat results
        results = []
        for adversary in self.adversaries:
            results.append(adversary(*args))

        return torch.cat(results, dim=1)
        

class AdversaryMixin(object):
    """
     TODO: I need to make this more generic and reusable
    """
    def initial_setup(self, args):
        super().initial_setup(args)
        if args.batch_size != 1:
            raise Exception("trainer requires batch_size 1")
        self.display_batch_size = 1
        self.img_shape = (args.resolution_width, args.resolution_height)

        if args.warmup_adversary_from > args.adversary_epoch_from:
            raise Exception("warmup_adversary_from > adversary_epoch_from. Bad configuration!")
        if not args.orientation_augmentation:
            raise Exception("Orientation augmentation required for adversary")

    def model_setup(self, args):
        super().model_setup(args)
        
        # Load adversary model as well
        logging.info("Loading adversary model")
        if args.adversary_model == "shallow":
            Adversary = Discriminator
        elif args.adversary_model == "wide":
            Adversary = WideDiscriminator
        elif args.adversary_model == "simple":
            Adversary = SimpleDiscriminator
        elif args.adversary_model == "unpooled_resnet50":
            Adversary = UnpooledResnet50
        elif args.adversary_model == "resnet50":
            Adversary = Resnet50
        else:
            raise Exception(f"Adversary model {args.adversary_model} not recognised")

        if args.number_of_adversaries > 1:
            adversary_model = AdversaryEnsemble(args.number_of_adversaries, 
                Adversary, num_classes=1, args=args)
        else:
            adversary_model = Adversary(num_classes=1, args=args)

        self.model = AdversaryModel(
            prediction_model=self.model, 
            adversary_model=adversary_model.cuda())

        nparams, nlayers = self.count_parameters(self.model.prediction_model)
        logging.info(f"Predictor parameters: {nparams:,} layers: {nlayers}")
        nparams, nlayers = self.count_parameters(self.model.adversary_model)
        logging.info(f"Adversary parameters: {nparams:,} layers: {nlayers}")

    def parameter_groups_setup(self, args):
        if args.dont_learn_predictor:
            prediction_scaling = 0.0
        else:
            prediction_scaling = 1.0

        self.parameter_groups = [
                    {'params': self.model.prediction_model.parameters(),
                     'group_scaling': prediction_scaling},
                    {'params': self.model.adversary_model.parameters(), 
                     'group_scaling': args.adversary_lr_scale,
                     'weight_decay': args.adversary_weight_decay}
        ]

    def predictor_adv_loss(self, prediction_reorien, true_label):
        bs, ch, h, w = prediction_reorien.shape
        args = self.args
        if self.runinfo["at_epoch"] >= args.adversary_epoch_from:
            ### Apply resnet
            toggle_grad(self.model.adversary_model, False)
            orientation_prediction = self.model(prediction_reorien, adversary=True)

            if args.adv_target_uncertain:
                false_label = 1 - true_label
            else:
                false_label = 0.5*torch.ones((bs, args.number_of_adversaries)).cuda()
            
            orien_loss_predictor = F.binary_cross_entropy_with_logits(orientation_prediction, false_label)
            orien_loss_predictor = orien_loss_predictor*args.adversary_strength
            #pdb.set_trace()
        else:
            orien_loss_predictor = torch.tensor(0.0).cuda()
        return orien_loss_predictor

    def adversary_adv_loss(self, prediction_reorien, true_label):
        args = self.args
        if self.runinfo["at_epoch"] >= args.warmup_adversary_from:
            # Encourage the adversary to predict the correct orientation
            toggle_grad(self.model.adversary_model, True)
            prediction_reorien_adv = prediction_reorien.detach()
            prediction_reorien_adv.requires_grad_() #TODO Might not be required
            orientation_prediction_adv = self.model(prediction_reorien_adv, adversary=True)
            orien_loss_adv = F.binary_cross_entropy_with_logits(orientation_prediction_adv, true_label)

            # Prediction error for logging
            correct = (orientation_prediction_adv>0).float() == true_label
            accuracy = correct.float().mean()
            #pdb.set_trace()

            if args.reg_param > 0:
                reg = args.reg_param * compute_grad2(
                    orientation_prediction_adv, 
                    prediction_reorien_adv).mean()
            else:
                reg = torch.tensor(0.0).cuda()
        else:
            reg = torch.tensor(0.0).cuda()
            orien_loss_adv = torch.tensor(0.0).cuda()
            accuracy = torch.tensor(0.0).cuda()
            orientation_prediction_adv = torch.tensor(0.0).cuda()
        return orien_loss_adv, orientation_prediction_adv.mean(), accuracy, reg

    def additional_training_loss_terms(self, loss_dict, batch, prediction, target):
        """
            Adds the orientation adversary terms to the loss_dict
        """
        args = self.args
        loss_dict, batch, prediction, target = super().additional_training_loss_terms(
                                                    loss_dict, batch, prediction, target)
        if prediction is None:
            raise Exception("AdversaryMixin only supports Trainers which output prediction from the training_loss")
        bs, ch, h, w = prediction.shape

        # Orientation augmentation
        is_rotated = batch['attrs_dict']['rotated'].cuda()
        prediction_rot = prediction.transpose(-2, -1)
        # If tagged as rotated then rotate (transpose) it back
        prediction_reorien = torch.where(is_rotated, prediction_rot, prediction)
        true_label = is_rotated.float()[..., None]
        true_label = true_label.expand(bs, args.number_of_adversaries)

        # change to three input channels so we can use standard resnet models
        prediction_reorien = prediction_reorien.expand(-1, 3, -1, -1)

        # Normalize to std 1 centered
        prediction_reorien = prediction_reorien - prediction_reorien.mean()
        prediction_reorien = prediction_reorien.div(prediction_reorien.std())

        orien_loss_predictor = self.predictor_adv_loss(prediction_reorien, true_label)
        orien_loss_adv, opred, accuracy, reg = self.adversary_adv_loss(prediction_reorien, true_label)

        pred_loss = loss_dict['train_loss']
        total_loss = pred_loss + orien_loss_predictor + orien_loss_adv + reg

        #pdb.set_trace()
        loss_dict = { **loss_dict,
            'train_loss': total_loss, 
            'pred_loss': pred_loss,
            'pred_orien': orien_loss_predictor,
            'accu_orien': accuracy,
            'adv_bias': opred,
            'adv_orien': orien_loss_adv,
            'reg': reg}

        if args.debug:
            # Monitor the contribution from the two loss terms to the predictor
            prediction_reorien.register_hook(lambda g: self.grad_log(g, "adversary -> predictor"))
            prediction.register_hook(lambda g: self.grad_log(g, "predictor + adversary"))
            orientation_prediction_adv.register_hook(lambda g: self.grad_log(g, "adversary"))
            reg.register_hook(lambda g: self.grad_log(g, "reg"))
            
        return loss_dict, batch, prediction, target

    def training_loss_hook(self, progress, losses, logging_epoch):
        super().training_loss_hook(progress, losses, logging_epoch)
        epoch = self.runinfo["at_epoch"]
        if epoch < self.args.warmup_adversary_from:
            return

        warm = (epoch >= self.args.warmup_adversary_from) and (epoch < self.args.adversary_epoch_from)

        if logging_epoch:
            logging.info(
                f"pred: {losses['instantaneous_pred_loss']:1.4f} "
                f"pred or: {losses['instantaneous_pred_orien']:1.4f} " 
                f"adv or: {losses['instantaneous_adv_orien']:1.4f} "
                f"acc: {losses['instantaneous_accu_orien']:1.0f} "
                f"bias: {losses['instantaneous_adv_bias']:1.2e} "
                f"reg: {losses['instantaneous_reg']:1.1e} warm: {warm}")
            logging.info(
                f"Running acc: {losses['average_accu_orien']:1.3f}")

    def grad_log(self, g, name):
        logging.info(f"    Norm of gradient from {name}: {g.norm().item():1.2e}")
        return g

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
