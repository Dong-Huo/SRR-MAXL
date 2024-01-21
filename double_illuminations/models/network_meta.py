import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import torch.nn.functional as F
from models.modules import *


class aggregate_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(aggregate_net, self).__init__()

        self.sensitivity_net = sensitivity_net(in_channels, out_channels * 3)
        self.pyramid_net = pyramid_net(in_channels, out_channels)
        self.auxiliary_net = auxiliary_net(in_channels, out_channels)

    def forward(self, rgb, illumination):
        b, c, h_inp, w_inp = rgb.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        rgb = F.pad(rgb, [0, pad_w, 0, pad_h], mode='reflect')

        sensitivity = self.sensitivity_net(rgb)
        ref_all, shared_feature = self.pyramid_net(rgb, illumination, sensitivity)
        rgb_back = self.auxiliary_net(shared_feature, ref_all[0])

        ref_all[0] = ref_all[0][:, :, :h_inp, :w_inp]

        rgb_back = rgb_back[:, :, :h_inp, :w_inp]

        return sensitivity, ref_all, rgb_back
