import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import pdb

kernel_size = 3

class SimpleDiscriminator(nn.Module):
    """
        Known to work well as a GAN discriminator
        
    """
    def __init__(self, num_classes=1, args=None):
        super().__init__()
        nf = self.nf = 128

        # Submodules
        nlayers = 0
        self.nf0 = nf

        blocks = [
            ResnetBlock(nf, nf),
            ResnetBlock(nf, nf),
        ]

        # Initial up-channeling conv
        self.conv_img = nn.Conv2d(3, 1*nf, kernel_size=kernel_size, padding=kernel_size//2)

        self.resnet = nn.Sequential(*blocks)

        # Final stage is standard avg-pool followed by linear
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.nf0, num_classes)
        #self.norm = nn.GroupNorm(1, 1, affine=False, eps=0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        
        #out = self.norm(out)
        #pdb.set_trace()
        out = self.conv_img(out)
        out = self.resnet(out)
        out = self.pool(out)
        out = out.view(batch_size, self.nf0)
        out = self.fc(actvn(out))
        
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.norm_0 = nn.GroupNorm(self.fin//32, self.fin)

        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 
            kernel_size, stride=1, padding=kernel_size//2, bias=False)

        self.norm_1 = nn.GroupNorm(self.fhidden//32, self.fhidden)

        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 
            kernel_size, stride=1, padding=kernel_size//2, bias=False)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 
                1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.norm_0(x)))
        dx = self.conv_1(actvn(self.norm_1(dx)))
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    return F.relu(x)
    #return F.leaky_relu(x, 2e-1)
