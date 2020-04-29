import torch
import torch.optim as optim
import torch.nn as nn
from RAKI.RAKI_model import RAKI
import RAKI.Utilities as utils
import data.transforms as transforms
import torch.nn.functional as F


class RAKI_trainer:
    def __init__(self, accelaration_rate, epochs=100):
        self.model = RAKI(accelaration_rate)
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
        optimizer = optim.Adam([{'params': first_layer_params, 'lr': '100'}, {'params': remaining_params}, ], lr=10)
        # setup data for training
        real_masked_kspace = transforms.complex_to_chans(masked_input_kspace)
        # train model
        self.model.train()
        for i in range(self.epochs):
            optimizer.zero_grad()
            reconstructed_lines = self.model(real_masked_kspace)
            loss = self.calculate_loss(reconstructed_lines, masked_input_kspace, ref_kspace, mask)
            print("RAKI loss at epoch %d: %d", i, loss)
            loss.backward()
            optimizer.step()

        self.model.eval()
        return torch.where(mask == 0, transforms.chans_to_complex(self.model(real_masked_kspace)), masked_input_kspace)

    def initialize_conv_weights(self, module):
        if type(module) == nn.Conv2d:
            module.weight.data = utils.truncated_normal(torch.zeros(module.weight.size(), dtype=torch.float32,
                                                                    requires_grad=True), std=0.1)

    def multicoil_to_combined_kspace(self, k_space):
        image = transforms.ifft2(k_space)
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares
        image = transforms.root_sum_of_squares(image, 1)
        return transforms.fft2(image)

    def calculate_loss(self, reconstructed_lines, masked_kspace, ref_kspace, mask):
        combined_ref_kspace = self.multicoil_to_combined_kspace(ref_kspace)
        reconstructed_kspace = torch.where(mask == 0, transforms.chans_to_complex(reconstructed_lines), masked_kspace)
        combined_reconstructed_kspace = self.multicoil_to_combined_kspace(reconstructed_kspace)
        return F.mse_loss(combined_reconstructed_kspace, combined_ref_kspace)
