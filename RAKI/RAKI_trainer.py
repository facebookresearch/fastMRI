import torch
import torch.optim as optim
import torch.nn as nn
from RAKI.RAKI_model import RAKI
import RAKI.Utilities as utils
import data.transforms as transforms
import torch.nn.functional as F
import numpy as np


class RAKI_trainer:
    def __init__(self, acceleration_rate, epochs=100):
        self.kx_1 = 5 
        self.ky_1 = 2
        self.kx_2 = 1
        self.ky_2 = 1
        self.kx_3 = 3
        self.ky_3 = 2
        self.model = RAKI(self.kx_1, self.ky_1, self.kx_2, self.ky_2, self.kx_3, self.ky_3, acceleration_rate.item())
        self.model.to(acceleration_rate.device)
        self.epochs = epochs
        self.acceleration_rate = acceleration_rate


    def train(self, masked_input_kspace, ref_kspace, mask):
        # initialize weights based on a truncated normal distribution
        self.model.apply(self.initialize_conv_weights)
        # Set the learning rate of first conv layer to 100 and the remaining conv layers to 10
        first_layer = ['model.0.weight', 'model.0.bias']
        first_layer_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in first_layer,
                                                                  self.model.named_parameters()))))
        remaining_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in first_layer,
                                                                self.model.named_parameters()))))
        optimizer = optim.Adam([{'params': first_layer_params, 'lr': 0.001}, {'params': remaining_params}, ], lr=0.0001)
        # setup data for training
        real_masked_kspace = transforms.complex_to_chans(masked_input_kspace)
        #real_masked_kspaces = torch.cat([real_masked_kspace, real_masked_kspace], dim=0)
        real_masked_kspaces = real_masked_kspace.repeat(30, 1, 1, 1)
        # train model
        with torch.enable_grad():
            self.model.train()
            for i in range(self.epochs):
                optimizer.zero_grad()
                reconstructed_lines = self.model(real_masked_kspaces)
                loss = F.mse_loss(self.reconstruct_kspace(real_masked_kspace, reconstructed_lines), transforms.complex_to_chans(ref_kspace).squeeze(0))
                if i == self.epochs - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                optimizer.step()

        self.model.eval()
        reconstructed_kspace = self.reconstruct_kspace(real_masked_kspace, self.model(real_masked_kspaces)).unsqueeze(0)
        return reconstructed_kspace.detach()

    def initialize_conv_weights(self, module):
        if type(module) == nn.Conv2d:
            module.weight.data = utils.truncated_normal(torch.zeros(module.weight.size(), requires_grad=True, dtype=torch.float32, device=torch.device('cuda:0')), std=0.1)

    def reconstruct_kspace(self, input_real_kspace, reconstructed_lines):
        kspace_recon = input_real_kspace.clone().squeeze(0)
        target_x_start = np.int32(np.ceil(self.kx_1/2) + np.floor(self.kx_2/2) + np.floor(self.kx_3/2) -1)
        target_x_end_kspace = input_real_kspace.size(-2) - target_x_start
        for ind_acc in range(0, self.acceleration_rate.item() - 1):
            target_y_start = np.int32((np.ceil(self.ky_1/2)-1) + np.int32((np.ceil(self.ky_2/2)-1)) + np.int32(np.ceil(self.ky_3/2)-1)) * self.acceleration_rate.item() + ind_acc + 1             
            target_y_end_kspace = input_real_kspace.size(-1) - np.int32((np.floor(self.ky_1/2)) + (np.floor(self.ky_2/2)) + np.floor(self.ky_3/2)) * self.acceleration_rate.item() + ind_acc
            indexed_recon_lines = reconstructed_lines[:, ind_acc, :, ::self.acceleration_rate.item()]
            kspace_recon[:,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:self.acceleration_rate.item()] = indexed_recon_lines
        return kspace_recon

