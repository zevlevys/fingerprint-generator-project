import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_SE
from models.stylegan2.model import EqualLinear


class MntToVecEncoderEncoderIntoW(Module):
    def __init__(self, opts=None):
        super(MntToVecEncoderEncoderIntoW, self).__init__()
        blocks = get_blocks(num_layers=50)
        unit_module = bottleneck_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x
