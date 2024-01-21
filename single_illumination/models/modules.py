import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from models.Transblock import MSAB


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(conv, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        out = self.stem(x)
        return out


class conv_norelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(conv_norelu, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias))

    def forward(self, x):
        out = self.stem(x)
        return out


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dilation=1, bias=False):
        super(upconv, self).__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=0, output_padding=0, bias=bias),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        out = self.stem(x)
        return out


class upconv_norelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dilation=1, bias=False):
        super(upconv_norelu, self).__init__()
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=0, output_padding=0, bias=bias),

        )

    def forward(self, x):
        out = self.stem(x)
        return out


class resnet_block(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1, bias=False):
        super(resnet_block, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out


class GLP_block(nn.Module):
    def __init__(self, feature_channels, hyper_channels):
        super(GLP_block, self).__init__()

        self.feature_conv1_0 = conv_norelu(feature_channels, hyper_channels, kernel_size=1, stride=1)

        self.feature_conv1_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv1_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv1_3 = conv_norelu(hyper_channels, hyper_channels, kernel_size=4, stride=2)

        self.feature_conv2_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv2_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv2_3 = upconv_norelu(hyper_channels, hyper_channels)

        self.hyper_conv1_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.hyper_conv1_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.hyper_conv1_3 = upconv_norelu(hyper_channels, hyper_channels)

        self.mask_conv1_1 = conv(hyper_channels * 3, hyper_channels, kernel_size=3, stride=1)
        self.mask_conv1_2 = conv(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.mask_conv1_3 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)

    def forward(self, feature, hyper_output):
        feature = self.feature_conv1_0(feature)
        down_feature = self.feature_conv1_3(self.feature_conv1_2(self.feature_conv1_1(feature)))
        up_feature = self.feature_conv2_3(self.feature_conv2_2(self.feature_conv2_1(down_feature)))
        # up_feature = self.feature_crop(up_feature, feature)

        up_hyper = self.hyper_conv1_3(self.hyper_conv1_2(self.hyper_conv1_1(hyper_output)))
        # up_hyper = self.feature_crop(up_hyper, feature)

        mask = self.mask_conv1_3(self.mask_conv1_2(self.mask_conv1_1(torch.cat([feature, up_feature, up_hyper], 1))))

        return up_hyper + mask * (feature - up_feature)


class out_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(out_block, self).__init__()

        self.conv1 = resnet_block(in_channels, kernel_size=kernel_size)
        self.conv2 = resnet_block(in_channels, kernel_size=kernel_size)
        self.conv3 = resnet_block(in_channels, kernel_size=kernel_size)

        self.conv4 = conv_norelu(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, feature):
        return self.conv4(self.conv3(self.conv2(self.conv1(feature))))


class attention_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, kernel_size=3):
        super(attention_block, self).__init__()

        self.conv1 = conv(in_channels * 2, intermediate_channels, kernel_size=kernel_size)

        self.conv2 = resnet_block(intermediate_channels, kernel_size=kernel_size)
        self.conv3 = resnet_block(intermediate_channels, kernel_size=kernel_size)
        self.conv4 = resnet_block(intermediate_channels, kernel_size=kernel_size)

        self.conv5 = conv_norelu(intermediate_channels, 1, kernel_size=kernel_size)

    def forward(self, ref, delta_ref):
        # mask = F.relu(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(torch.cat([ref, delta_ref], 1)))))),
        #               inplace=True)

        mask = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(torch.cat([ref, delta_ref], 1))))))

        # return ref * mask[:, 0:1, :, :] + delta_ref
        return delta_ref


class auxiliary_net(nn.Module):
    def __init__(self, out_channels, ref_channel):
        super(auxiliary_net, self).__init__()

        ks = 3
        ch1 = 31

        self.conv1 = conv_norelu(ch1, ch1, kernel_size=ks, stride=1)
        self.conv2 = conv_norelu(ch1 + ref_channel, ch1, kernel_size=ks, stride=1)
        self.conv3 = resnet_block(ch1, kernel_size=ks)
        self.conv4 = resnet_block(ch1, kernel_size=ks)
        self.conv5 = resnet_block(ch1, kernel_size=ks)

        self.conv6 = resnet_block(ch1, kernel_size=ks)
        self.conv7 = resnet_block(ch1, kernel_size=ks)
        self.conv8 = resnet_block(ch1, kernel_size=ks)

        self.conv9 = conv_norelu(ch1, out_channels, kernel_size=ks, stride=1)

    def forward(self, shared_feature, refined_ref):
        out = self.conv1(shared_feature)
        out = self.conv2(torch.cat([out, refined_ref], 1))
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)

        rgb_back = self.conv9(out)

        return rgb_back


class sensitivity_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(sensitivity_net, self).__init__()

        ks = 3
        ch1 = 31
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        ch4 = ch3 * 2

        self.num_alpha = in_channels // 3
        self.out_channels = out_channels

        # scale = pow((out_channels / in_channels), 1 / num_steps)
        # self.channel_sizes = [round(in_channels * pow(scale, i)) for i in range(num_steps)]
        # self.channel_sizes.append(out_channels)

        # ----------------pre-processing----------------

        # self.camera_sensitivity = torch.from_numpy(camera_sensitivity).view(1, -1, 31).float().cuda()

        # ----------------encoder----------------
        # scale 1/1
        self.en_conv0_1 = conv_norelu(in_channels // self.num_alpha, ch1, kernel_size=ks, stride=1)
        # self.en_conv0_2 = conv_norelu(in_channels // self.num_alpha, ch1, kernel_size=ks, stride=1)

        self.en_conv1_1 = conv_norelu(ch1 * self.num_alpha, ch1, kernel_size=ks, stride=1)
        self.en_conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_4 = resnet_block(ch1, kernel_size=ks)

        # scale 1/2
        self.en_conv2_1 = conv(ch1, ch2, kernel_size=4, stride=2)
        self.en_conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_4 = resnet_block(ch2, kernel_size=ks)

        # scale 1/4
        self.en_conv3_1 = conv(ch2, ch3, kernel_size=4, stride=2)
        self.en_conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_4 = resnet_block(ch3, kernel_size=ks)

        # scale 1/8
        self.en_conv4_1 = conv(ch3, ch4, kernel_size=4, stride=2)
        self.en_conv4_2 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_3 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_4 = resnet_block(ch4, kernel_size=ks)

        # # scale 1/16
        # self.en_conv5_1 = conv(ch4, ch5, kernel_size=ks, stride=2)
        # self.en_conv5_2 = resnet_block(ch5, kernel_size=ks)
        # self.en_conv5_3 = resnet_block(ch5, kernel_size=ks)
        # self.en_conv5_4 = resnet_block(ch5, kernel_size=ks)
        #
        # # scale 1/32
        # self.en_conv6_1 = conv(ch5, ch6, kernel_size=ks, stride=2)
        # self.en_conv6_2 = resnet_block(ch6, kernel_size=ks)
        # self.en_conv6_3 = resnet_block(ch6, kernel_size=ks)
        # self.en_conv6_4 = resnet_block(ch6, kernel_size=ks)

        # output
        self.out_conv1_1 = conv_norelu(ch4, self.out_channels, kernel_size=1, stride=1)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # self.pc = torch.from_numpy(np.load("datasets/pc.npy")).float()  # 3, 31, 2 rgb

    def forward(self, rgb):
        # multi-rgb fusion

        # self.pc = self.pc.to(rgb.device)

        conv0_1 = self.en_conv0_1(rgb[:, 0:3, :, :])
        # conv0_2 = self.en_conv0_2(rgb[:, 3:6, :, :])

        # scale 1/1
        conv1_d = self.en_conv1_1(conv0_1)
        conv1_d = self.en_conv1_4(self.en_conv1_3(self.en_conv1_2(conv1_d)))

        # scale 1/2
        conv2_d = self.en_conv2_1(conv1_d)
        conv2_d = self.en_conv2_4(self.en_conv2_3(self.en_conv2_2(conv2_d)))

        # scale 1/4
        conv3_d = self.en_conv3_1(conv2_d)
        conv3_d = self.en_conv3_4(self.en_conv3_3(self.en_conv3_2(conv3_d)))

        # scale 1/8
        conv4_d = self.en_conv4_1(conv3_d)
        conv4_d = self.en_conv4_4(self.en_conv4_3(self.en_conv4_2(conv4_d)))

        # # scale 1/16
        # conv5_d = self.en_conv5_1(conv4_d)
        # conv5_d = self.en_conv5_4(self.en_conv5_3(self.en_conv5_2(conv5_d)))
        #
        # # scale 1/32
        # conv6_d = self.en_conv6_1(conv5_d)
        # conv6_d = self.en_conv6_4(self.en_conv6_3(self.en_conv6_2(conv6_d)))

        # output
        sensitivity = self.out_conv1_1(conv4_d)

        # b, 3, 2
        sensitivity = self.pooling(sensitivity).view(-1, 3, self.out_channels // 3)

        # sensitivity = torch.einsum("bnk,njk->bnj", sensitivity_weights.to(torch.float64),
        #                            self.pc.to(torch.float64)).to(torch.float32)

        return sensitivity


class pyramid_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pyramid_net, self).__init__()

        # ----------------configuration----------------
        ks = 3
        ch1 = 31
        ch2 = ch1 * 2
        ch3 = ch2 * 2
        ch4 = ch3 * 2
        ch5 = ch4 * 2
        ch6 = ch5 * 2

        ch_list = [ch1, ch2, ch3, ch4, ch5, ch6]

        self.num_alpha = in_channels // 3
        self.out_channels = out_channels

        # scale = pow((out_channels / in_channels), 1 / num_steps)
        # self.channel_sizes = [round(in_channels * pow(scale, i)) for i in range(num_steps)]
        # self.channel_sizes.append(out_channels)

        # ----------------pre-processing----------------

        # self.camera_sensitivity = torch.from_numpy(camera_sensitivity).view(1, -1, 31).float().cuda()

        # ----------------encoder----------------
        # scale 1/1
        self.en_conv0_1 = conv_norelu(in_channels // self.num_alpha, ch1, kernel_size=ks, stride=1)
        # self.en_conv0_2 = conv_norelu(in_channels // self.num_alpha, ch1, kernel_size=ks, stride=1)

        self.en_conv1_1 = conv_norelu(ch1 * self.num_alpha, ch1, kernel_size=ks, stride=1)
        self.en_conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_4 = resnet_block(ch1, kernel_size=ks)

        # scale 1/2
        self.en_conv2_1 = conv(ch1, ch2, kernel_size=4, stride=2)
        self.en_conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_4 = resnet_block(ch2, kernel_size=ks)

        # scale 1/4
        self.en_conv3_1 = conv(ch2, ch3, kernel_size=4, stride=2)
        self.en_conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_4 = resnet_block(ch3, kernel_size=ks)

        # scale 1/8
        self.en_conv4_1 = conv(ch3, ch4, kernel_size=4, stride=2)
        self.en_conv4_2 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_3 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_4 = resnet_block(ch4, kernel_size=ks)

        # # scale 1/16
        # self.en_conv5_1 = conv(ch4, ch5, kernel_size=ks, stride=2)
        # self.en_conv5_2 = resnet_block(ch5, kernel_size=ks)
        # self.en_conv5_3 = resnet_block(ch5, kernel_size=ks)
        # self.en_conv5_4 = resnet_block(ch5, kernel_size=ks)
        #
        # # scale 1/32
        # self.en_conv6_1 = conv(ch5, ch6, kernel_size=ks, stride=2)
        # self.en_conv6_2 = resnet_block(ch6, kernel_size=ks)
        # self.en_conv6_3 = resnet_block(ch6, kernel_size=ks)
        # self.en_conv6_4 = resnet_block(ch6, kernel_size=ks)

        # ----------------decoder----------------

        # # scale 1/16
        # self.de_conv5_1_1 = upconv(ch6, ch5)
        # self.de_conv5_1_2 = GLPBlock(ch5, out_channels)
        #
        # self.de_conv5_c = conv_norelu(ch5 * 2 + self.out_channels, ch5, kernel_size=1, stride=1)
        # self.de_conv5_2 = resnet_block(ch5, kernel_size=ks)
        # self.de_conv5_3 = resnet_block(ch5, kernel_size=ks)
        # self.de_conv5_4 = resnet_block(ch5, kernel_size=ks)
        #
        # # scale 1/8
        # self.de_conv4_1_1 = upconv(ch5, ch4)
        # self.de_conv4_1_2 = GLPBlock(ch4, out_channels)
        #
        # self.de_conv4_c = conv_norelu(ch4 * 2 + self.out_channels, ch4, kernel_size=1, stride=1)
        # self.de_conv4_2 = resnet_block(ch4, kernel_size=ks)
        # self.de_conv4_3 = resnet_block(ch4, kernel_size=ks)
        # self.de_conv4_4 = resnet_block(ch4, kernel_size=ks)

        self.trans = MSAB(ch4, ch1, ch4 // ch1, 4)

        # scale 1/4
        self.de_conv3_1_1 = upconv(ch4, ch3)
        self.de_conv3_1_2 = GLP_block(ch3, out_channels)

        self.de_conv3_c = conv_norelu(ch3 * 2 + self.out_channels, ch3, kernel_size=1, stride=1)
        self.de_conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.de_conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.de_conv3_4 = resnet_block(ch3, kernel_size=ks)

        # scale 1/2
        self.de_conv2_1_1 = upconv(ch3, ch2)
        self.de_conv2_1_2 = GLP_block(ch2, out_channels)

        self.de_conv2_c = conv_norelu(ch2 * 2 + self.out_channels, ch2, kernel_size=1, stride=1)
        self.de_conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.de_conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.de_conv2_4 = resnet_block(ch2, kernel_size=ks)

        # scale 1/1
        self.de_conv1_1_1 = upconv(ch2, ch1)
        self.de_conv1_1_2 = GLP_block(ch1, out_channels)

        self.de_conv1_c = conv_norelu(ch1 * 2 + self.out_channels, ch1, kernel_size=1, stride=1)
        self.de_conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.de_conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.de_conv1_4 = resnet_block(ch1, kernel_size=ks)

        # output layers

        # self.out_conv6 = conv_norelu(ch_list[5], out_channels - 3, kernel_size=ks)
        # self.out_conv5 = conv_norelu(ch_list[4], out_channels - 3, kernel_size=ks)

        # self.out_null_conv4 = out_block(ch_list[3], out_channels - 3 * self.num_alpha, kernel_size=ks)
        # self.out_null_conv3 = out_block(ch_list[2], out_channels - 3 * self.num_alpha, kernel_size=ks)
        # self.out_null_conv2 = out_block(ch_list[1], out_channels - 3 * self.num_alpha, kernel_size=ks)
        # self.out_null_conv1 = out_block(ch_list[0], out_channels - 3 * self.num_alpha, kernel_size=ks)

        self.out_delta_conv4 = out_block(ch_list[3], out_channels, kernel_size=ks)
        self.out_delta_conv3 = out_block(ch_list[2], out_channels, kernel_size=ks)
        self.out_delta_conv2 = out_block(ch_list[1], out_channels, kernel_size=ks)
        self.out_delta_conv1 = out_block(ch_list[0], out_channels, kernel_size=ks)

        self.attention_block4 = attention_block(out_channels, ch_list[3], kernel_size=ks)
        self.attention_block3 = attention_block(out_channels, ch_list[2], kernel_size=ks)
        self.attention_block2 = attention_block(out_channels, ch_list[1], kernel_size=ks)
        self.attention_block1 = attention_block(out_channels, ch_list[0], kernel_size=ks)

        # # refinement layers
        #
        # self.re_conv1 = conv(out_channels, ch1, kernel_size=ks, stride=1)
        # self.re_conv2 = resnet_block(ch1, kernel_size=ks)
        # self.re_conv3 = resnet_block(ch1, kernel_size=ks)
        # self.re_conv4 = resnet_block(ch1, kernel_size=ks)
        # self.re_out_conv = conv_norelu(ch1, out_channels, kernel_size=ks, stride=1)

    def forward(self, rgb, illumination, sensitivity):
        ref_all = []

        # multi-rgb fusion
        conv0_1 = self.en_conv0_1(rgb[:, 0:3, :, :])
        # conv0_2 = self.en_conv0_2(rgb[:, 3:6, :, :])

        # scale 1/1
        conv1_d = self.en_conv1_1(conv0_1)
        conv1_d = self.en_conv1_4(self.en_conv1_3(self.en_conv1_2(conv1_d)))

        # scale 1/2
        conv2_d = self.en_conv2_1(conv1_d)
        conv2_d = self.en_conv2_4(self.en_conv2_3(self.en_conv2_2(conv2_d)))

        # scale 1/4
        conv3_d = self.en_conv3_1(conv2_d)
        conv3_d = self.en_conv3_4(self.en_conv3_3(self.en_conv3_2(conv3_d)))

        # scale 1/8
        conv4_d = self.en_conv4_1(conv3_d)
        conv4_d = self.en_conv4_4(self.en_conv4_3(self.en_conv4_2(conv4_d)))

        # # scale 1/16
        # conv5_d = self.en_conv5_1(conv4_d)
        # conv5_d = self.en_conv5_4(self.en_conv5_3(self.en_conv5_2(conv5_d)))
        #
        # # scale 1/32
        # conv6_d = self.en_conv6_1(conv5_d)
        # conv6_d = self.en_conv6_4(self.en_conv6_3(self.en_conv6_2(conv6_d)))
        #
        # ref_null = self.out_conv6(conv6_d)
        # ref = self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, ref_null)
        # ref_all.insert(0, ref)

        # # scale 1/16
        # conv5_skip = self.de_conv5_1_1(conv6_d)
        # ref = self.de_conv5_1_2(conv5_d, ref)
        # conv5_skip = self.feature_crop(conv5_skip, conv5_d)
        #
        # conv5_d = self.de_conv5_c(torch.cat([conv5_skip, conv5_d, ref], 1))
        # conv5_d = self.de_conv5_4(self.de_conv5_3(self.de_conv5_2(conv5_d)))
        #
        # ref_null = self.out_conv5(conv5_d)
        # ref = self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, ref_null)
        # ref_all.insert(0, ref)
        #
        # # scale 1/8
        # conv4_skip = self.de_conv4_1_1(conv5_d)
        # ref = self.de_conv4_1_2(conv4_d, ref)
        # conv4_skip = self.feature_crop(conv4_skip, conv4_d)
        #
        # conv4_d = self.de_conv4_c(torch.cat([conv4_skip, conv4_d, ref], 1))
        # conv4_d = self.de_conv4_4(self.de_conv4_3(self.de_conv4_2(conv4_d)))
        #

        conv4_d = self.trans(conv4_d)

        basic_vectors_transpose, temp, null_vectors = self.sub_space(sensitivity, illumination)

        # ref_null = self.out_null_conv4(conv4_d)
        delta_ref = self.out_delta_conv4(conv4_d)
        ref = self.attention_block4(self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, conv4_d),
                                    delta_ref)
        ref_all.insert(0, ref)

        # scale 1/4
        conv3_skip = self.de_conv3_1_1(conv4_d)
        ref = self.de_conv3_1_2(conv3_d, ref)
        # conv3_skip = self.feature_crop(conv3_skip, conv3_d)

        conv3_d = self.de_conv3_c(torch.cat([conv3_skip, conv3_d, ref], 1))
        conv3_d = self.de_conv3_4(self.de_conv3_3(self.de_conv3_2(conv3_d)))

        # ref_null = self.out_null_conv3(conv3_d)
        delta_ref = self.out_delta_conv3(conv3_d)
        ref = self.attention_block3(self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, conv3_d),
                                    delta_ref)
        ref_all.insert(0, ref)

        # scale 1/2
        conv2_skip = self.de_conv2_1_1(conv3_d)
        ref = self.de_conv2_1_2(conv2_d, ref)
        # conv2_skip = self.feature_crop(conv2_skip, conv2_d)

        conv2_d = self.de_conv2_c(torch.cat([conv2_skip, conv2_d, ref], 1))
        conv2_d = self.de_conv2_4(self.de_conv2_3(self.de_conv2_2(conv2_d)))

        # ref_null = self.out_null_conv2(conv2_d)
        delta_ref = self.out_delta_conv2(conv2_d)
        ref = self.attention_block2(self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, conv2_d),
                                    delta_ref)
        ref_all.insert(0, ref)

        # scale 1/1
        conv1_skip = self.de_conv1_1_1(conv2_d)
        ref = self.de_conv1_1_2(conv1_d, ref)
        # conv1_skip = self.feature_crop(conv1_skip, conv1_d)

        conv1_d = self.de_conv1_c(torch.cat([conv1_skip, conv1_d, ref], 1))

        conv1_d = self.de_conv1_4(self.de_conv1_3(self.de_conv1_2(conv1_d)))

        shared_feature = conv1_d

        # ref_null = self.out_null_conv1(conv1_d)
        delta_ref = self.out_delta_conv1(conv1_d)
        ref = self.attention_block1(self.ref_generation(basic_vectors_transpose, temp, null_vectors, rgb, conv1_d),
                                    delta_ref)
        ref_all.insert(0, ref)

        # refined_out = self.re_conv1(ref)
        # refined_out = self.re_conv2(refined_out)
        # refined_out = self.re_conv3(refined_out)
        # refined_out = self.re_conv4(refined_out)
        # refined_ref = self.re_out_conv(refined_out) + ref

        return ref_all, shared_feature

    def sub_space(self, sensitivity, illumination):
        b = sensitivity.shape[0]
        illumination = illumination.view(b, self.num_alpha, 1, -1).repeat(1, 1, 3, 1)
        sensitivity = sensitivity.view(b, 1, 3, -1).repeat(1, self.num_alpha, 1, 1)

        # calculate basic vectors of subspace and projection matrix
        basic_vectors_transpose = (sensitivity * illumination).view(b, 3 * self.num_alpha, -1)
        basic_vectors = basic_vectors_transpose.permute(0, 2, 1).contiguous()

        # inverse_matrix = torch.linalg.inv(
        #     torch.einsum("bij,bjk->bik", basic_vectors_transpose.to(torch.float64),
        #                  basic_vectors.to(torch.float64))).float()
        #
        # temp = torch.einsum("bij,bjk->bik", basic_vectors.to(torch.float64), inverse_matrix.to(torch.float64)).float()

        temp = torch.linalg.pinv(basic_vectors_transpose)

        # print(torch.max(temp))
        # print(torch.min(temp))
        # print(torch.mean(temp))

        subspace = torch.einsum("bij,bjk->bik", temp.to(torch.float64),
                                basic_vectors_transpose.to(torch.float64)).float()

        nullspace = torch.eye(self.out_channels, device=subspace.device).unsqueeze(0).repeat(b, 1, 1) - subspace

        null_vectors, _, _ = torch.linalg.svd(nullspace)

        null_vectors = null_vectors.detach()

        null_vectors = null_vectors[:, :, :self.out_channels - 3 * self.num_alpha]

        return basic_vectors_transpose, temp, null_vectors

    # def feature_crop(self, feature_skip, feature_d):
    #
    #     if feature_skip.shape[2] > feature_d.shape[2] or feature_skip.shape[3] > feature_d.shape[3]:
    #         h_diff = feature_skip.shape[2] - feature_d.shape[2]
    #         w_diff = feature_skip.shape[3] - feature_d.shape[3]
    #         _, _, h, w = feature_d.shape
    #         feature_skip = feature_skip[:, :, h_diff // 2:h_diff // 2 + h, w_diff // 2:w_diff // 2 + w]
    #
    #     return feature_skip

    def ref_generation(self, basic_vectors_transpose, temp, null_vectors, rgb, feature):
        scaled_rgb = rgb

        if feature.shape[2] != scaled_rgb.shape[2] or feature.shape[3] != scaled_rgb.shape[3]:
            scaled_rgb = F.interpolate(scaled_rgb, size=(feature.shape[2], feature.shape[3]), mode='bilinear')

        ref_parallel = torch.einsum('bnc,bcij->bnij', temp.to(torch.float64),
                                    scaled_rgb.to(torch.float64)).float()

        # ref = ref_parallel + torch.einsum('bnc,bcij->bnij', null_vectors.to(torch.float64),
        #                                   ref_null.to(torch.float64)).float()

        ref = ref_parallel

        # rgb_back = torch.einsum('bnc,bcij->bnij', basic_vectors_transpose.to(torch.float64),
        #                             ref.to(torch.float64)).float()

        return ref
