import torch
import torch.optim as optim
import torch.nn as nn
from RAKI.RAKI_model import RAKI
import RAKI.Utilities as utils
import data.transforms as transforms
import torch.nn.functional as F


class RAKI_trainer:
    def __init__(self, acceleration_rate, epochs=200):
        self.model = RAKI(acceleration_rate.item())
        self.model.to(acceleration_rate.device)
        self.epochs = epochs

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
                loss = self.calculate_loss(reconstructed_lines, masked_input_kspace, ref_kspace, mask)
                print("RAKI loss at epoch {}: {}".format(i, loss))
                loss.backward(retain_graph=True)
                optimizer.step()

        self.model.eval()
         
        prediction = self.model(real_masked_kspaces)
        prediction = torch.where(mask == 0, prediction, ref_kspace)
        return torch.where(mask == 0, transforms.chans_to_complex(self.model(real_masked_kspace)), masked_input_kspace)

    def initialize_conv_weights(self, module):
        if type(module) == nn.Conv2d:
            module.weight.data = utils.truncated_normal(torch.zeros(module.weight.size(), requires_grad=True, dtype=torch.float32, device=torch.device('cuda:0')), std=0.1)

    def reconstruct_kspace(self, prediction, ref_ksp, mask):
        real_ref_ksp = transforms.complex_to_chans(ref_ksp).squeeze(0)
        reconstruct_filled = 

    def multicoil_to_combined_kspace(self, k_space):
        image = transforms.ifft2(k_space)
        # Apply Root-Sum-of-Squares
        image = transforms.root_sum_of_squares(image, 1)
        return transforms.fft2(image)

    def calculate_loss(self, reconstructed_lines, masked_kspace, ref_kspace, mask):
        #combined_ref_kspace = self.multicoil_to_combined_kspace(ref_kspace)
        real_ref_kspace = transforms.complex_to_chans(ref_kspace)
        target_line = torch.where(mask.squeeze(0) == 0, real_ref_kspace, torch.zeros_like(real_ref_kspace)).squeeze(0).unsqueeze(1)
        #target_line = torch.where(mask == 0, combined_ref_kspace, torch.zeros_like(combined_ref_kspace))
        #target_line = transforms.complex_to_chans(target_line).squeeze(0).unsqueeze(1)
        target_line = transforms.center_crop(target_line, (reconstructed_lines.size(-2), reconstructed_lines.size(-1)))
        #reconstructed_kspace = torch.where(mask == 0, transforms.chans_tcomplex(reconstructed_lines), masked_kspace)
        #combined_reconstructed_kspace = self.multicoil_to_combined_kspace(reconstructed_kspace)
        return F.mse_loss(reconstructed_lines, target_line)
