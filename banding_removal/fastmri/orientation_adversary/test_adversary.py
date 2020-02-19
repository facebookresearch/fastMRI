import sys
sys.path.append(sys.path[0] + "/../..")

import logging
import pdb
import cmath
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from fastmri.data import transforms
from fastmri.orientation_adversary.adversary_mixin import toggle_grad, compute_grad2

class ReferenceTrainer(object):
    """
        This is a simplification of a known good implementation
    """
    def __init__(self, generator, discriminator, use_reg=False):
        self.generator = generator
        self.discriminator = discriminator
        self.use_reg = use_reg

    def generator_loss(self, z):
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()

        x_fake = self.generator(z)

        d_fake = self.discriminator(x_fake)
        gloss = self.compute_loss(d_fake, 0)
        return gloss, x_fake

    def generator_trainstep(self, z):
        gloss, x_fakes = self.generator_loss(z)
        gloss.backward()
        return gloss.item()

    def discriminator_trainstep(self, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake)
        dloss = self.compute_loss(d_fake, 1)

        dloss.backward(retain_graph=True)

        if self.use_reg:
            reg = compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            reg = torch.tensor(0.)

        toggle_grad(self.discriminator, False)


        return dloss.item(), reg.item(), x_fake.detach()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

def test_adversary():
    """ The adversary mixin uses a different way of computing the loss
        compared to the reference implementation, it only calls backwards once.
        This unit test just compares this technique to a reference version.
    """
    use_reg = True
    nchan = 10
    bs = 1
    prediction_model = nn.Linear(nchan, nchan)
    adversary_model = nn.Linear(nchan, 1)
    prediction_model.train()
    adversary_model.train()

    data = torch.empty(bs, 10).normal_()
    data.requires_grad_()

    true_label = torch.ones((bs, 1))

    # Forward pass through prediction model
    prediction_reorien = prediction_model(data)

    ### Apply resnet
    toggle_grad(adversary_model, False)
    orientation_prediction = adversary_model(prediction_reorien)

    # Encourage the predictor to trick the adversary TODO: Should this be 0.5 instead?
    false_label = 1 - true_label
    orien_loss_predictor = F.binary_cross_entropy_with_logits(orientation_prediction, false_label)
    orien_loss_predictor = orien_loss_predictor

    # Encourage the adversary to predict the correct orientation
    toggle_grad(adversary_model, True)
    prediction_reorien_adv = prediction_reorien.detach()
    prediction_reorien_adv.requires_grad_() #TODO Might not be required
    orientation_prediction_adv = adversary_model(prediction_reorien_adv)
    orien_loss_adv = F.binary_cross_entropy_with_logits(orientation_prediction_adv, true_label)

    # Prediction error for logging
    correct = (orientation_prediction_adv>0).float() == true_label
    accuracy = correct.float().mean()

    if use_reg:
        reg = compute_grad2(
            orientation_prediction_adv, 
            prediction_reorien_adv).mean()
    else:
        reg = torch.zeros_like(orien_loss_adv)

    total_loss = orien_loss_predictor + orien_loss_adv + reg

    total_loss.backward()

    adversary_grad = adversary_model.weight.grad.clone()
    prediction_grad = prediction_model.weight.grad.clone()

    adversary_model.zero_grad()
    prediction_model.zero_grad()

    ####################################
    ### Now try reference implementation
    trainer = ReferenceTrainer(prediction_model, adversary_model, use_reg=use_reg)
    
    dloss, reg, _ = trainer.discriminator_trainstep(data)
    gloss = trainer.generator_trainstep(data)
    
    ref_adversary_grad = adversary_model.weight.grad.clone()
    ref_prediction_grad = prediction_model.weight.grad.clone()

    assert torch.allclose(adversary_grad, ref_adversary_grad)
    assert torch.allclose(prediction_grad, ref_prediction_grad)



