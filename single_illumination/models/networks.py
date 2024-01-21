import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                  padding=((kernel_size - 1) // 2) * dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def conv_weights(input, weights, name, kernel_size=3, stride=1, dilation=1):
    output = F.conv2d(input, weights['block{:d}.0.weight'.format(name)], weights['block{:d}.0.bias'.format(name)],
                      padding=((kernel_size - 1) // 2))

    output = F.leaky_relu(output, inplace=True)

    return output


def conv_norelu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                  padding=((kernel_size - 1) // 2) * dilation, bias=bias))


def conv_norelu_weights(input, weights, name, kernel_size=3, stride=1, dilation=1):
    output = F.conv2d(input, weights['block{:d}.0.weight'.format(name)], weights['block{:d}.0.bias'.format(name)],
                      padding=((kernel_size - 1) // 2))

    return output


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def upconv_weights(input, weights, name, kernel_size=3, stride=1, dilation=1):
    output = F.conv_transpose2d(input, weights['block{:d}.0.weight'.format(name)],
                                weights['block{:d}.0.bias'.format(name)], padding=1)

    output = F.leaky_relu(output, inplace=True)

    return output


def upconv_norelu(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
    )


def upconv_norelu_weights(input, weights, name, kernel_size=3, stride=1, dilation=1):
    output = F.conv_transpose2d(input, weights['block{:d}.0.weight'.format(name)],
                                weights['block{:d}.0.bias'.format(name)], padding=1)

    return output


def resnet_block(in_channels, kernel_size=3, dilation=[1, 1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)


def resnet_block_weights(input, weights, name, kernel_size=3, dilation=[1, 1], bias=True):
    output = F.conv2d(input, weights['block{:d}.0.weight'.format(name)], weights['block{:d}.0.bias'.format(name)],
                      padding=((kernel_size - 1) // 2))

    output = F.leaky_relu(output, inplace=True)

    output = F.conv2d(output, weights['block{:d}.0.weight'.format(name)], weights['block{:d}.0.bias'.format(name)],
                      padding=((kernel_size - 1) // 2))

    return output + input


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                      padding=((kernel_size - 1) // 2) * dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                      padding=((kernel_size - 1) // 2) * dilation[1], bias=bias),
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out


class GLPBlock(nn.Module):
    def __init__(self, feature_channels, hyper_channels):
        super(GLPBlock, self).__init__()

        self.feature_conv1_0 = conv_norelu(feature_channels, hyper_channels, kernel_size=1, stride=1)

        self.feature_conv1_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv1_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv1_3 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=2)

        self.feature_conv2_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv2_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.feature_conv2_3 = upconv_norelu(hyper_channels, hyper_channels)

        self.hyper_conv1_1 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.hyper_conv1_2 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.hyper_conv1_3 = upconv_norelu(hyper_channels, hyper_channels)

        self.mask_conv1_1 = conv(hyper_channels * 3, hyper_channels, kernel_size=3, stride=1)
        self.mask_conv1_2 = conv(hyper_channels, hyper_channels, kernel_size=3, stride=1)
        self.mask_conv1_3 = conv_norelu(hyper_channels, hyper_channels, kernel_size=3, stride=1)

    def forward(self, feature, hyper_output, weights=None):
        feature = self.feature_conv1_0(feature)
        down_feature = self.feature_conv1_3(self.feature_conv1_2(self.feature_conv1_1(feature)))
        up_feature = self.feature_conv2_3(self.feature_conv2_2(self.feature_conv2_1(down_feature)))
        up_feature = self.feature_crop(up_feature, feature)

        up_hyper = self.hyper_conv1_3(self.hyper_conv1_2(self.hyper_conv1_1(hyper_output)))
        up_hyper = self.feature_crop(up_hyper, feature)

        mask = self.mask_conv1_3(self.mask_conv1_2(self.mask_conv1_1(torch.cat([feature, up_feature, up_hyper], 1))))

        return up_hyper + mask * (feature - up_feature)

    def feature_crop(self, feature_skip, feature_d):
        if feature_skip.shape[2] > feature_d.shape[2] or feature_skip.shape[3] > feature_d.shape[3]:
            h_diff = feature_skip.shape[2] - feature_d.shape[2]
            w_diff = feature_skip.shape[3] - feature_d.shape[3]
            _, _, h, w = feature_d.shape
            feature_skip = feature_skip[:, :, h_diff // 2:h_diff // 2 + h, w_diff // 2:w_diff // 2 + w]

        return feature_skip


class pyramid_net(nn.Module):
    def __init__(self, in_channels, out_channels, illumination, camera_sensitivity):
        super(pyramid_net, self).__init__()

        # ----------------configuration----------------
        ks = 3
        ch1 = 32
        ch2 = 64
        ch3 = 128
        ch4 = 256
        ch5 = 512
        ch6 = 1024

        ch_list = [ch1, ch2, ch3, ch4, ch5, ch6]

        self.num_alpha = illumination.shape[0]
        self.out_channels = out_channels

        # scale = pow((out_channels / in_channels), 1 / num_steps)
        # self.channel_sizes = [round(in_channels * pow(scale, i)) for i in range(num_steps)]
        # self.channel_sizes.append(out_channels)

        # ----------------pre-processing----------------
        self.illumination = torch.from_numpy(illumination).view(1, -1, 31).float().cuda()
        self.camera_sensitivity = torch.from_numpy(camera_sensitivity).view(1, -1, 31).float().cuda()
        self.basic_vectors_transpose, self.temp, self.null_vectors = self.sub_space()

        # ----------------encoder----------------
        # scale 1/1
        self.en_conv0 = nn.Sequential(
            *[conv_norelu(in_channels // self.num_alpha, ch1, kernel_size=ks) for _ in range(self.num_alpha)])

        self.en_conv1_1 = conv_norelu(ch1 * self.num_alpha, ch1, kernel_size=ks, stride=1)
        self.en_conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.en_conv1_4 = resnet_block(ch1, kernel_size=ks)

        # scale 1/2
        self.en_conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.en_conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.en_conv2_4 = resnet_block(ch2, kernel_size=ks)

        # scale 1/4
        self.en_conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.en_conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.en_conv3_4 = resnet_block(ch3, kernel_size=ks)

        # scale 1/8
        self.en_conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.en_conv4_2 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_3 = resnet_block(ch4, kernel_size=ks)
        self.en_conv4_4 = resnet_block(ch4, kernel_size=ks)

        # scale 1/16
        self.en_conv5_1 = conv(ch4, ch5, kernel_size=ks, stride=2)
        self.en_conv5_2 = resnet_block(ch5, kernel_size=ks)
        self.en_conv5_3 = resnet_block(ch5, kernel_size=ks)
        self.en_conv5_4 = resnet_block(ch5, kernel_size=ks)

        # scale 1/32
        self.en_conv6_1 = conv(ch5, ch6, kernel_size=ks, stride=2)
        self.en_conv6_2 = resnet_block(ch6, kernel_size=ks)
        self.en_conv6_3 = resnet_block(ch6, kernel_size=ks)
        self.en_conv6_4 = resnet_block(ch6, kernel_size=ks)

        # ----------------decoder----------------

        # scale 1/16
        self.de_conv5_1_1 = upconv(ch6, ch5)
        self.de_conv5_1_2 = GLPBlock(ch5, out_channels)

        self.de_conv5_c = conv_norelu(ch5 * 2 + self.out_channels, ch5, kernel_size=1, stride=1)
        self.de_conv5_2 = resnet_block(ch5, kernel_size=ks)
        self.de_conv5_3 = resnet_block(ch5, kernel_size=ks)
        self.de_conv5_4 = resnet_block(ch5, kernel_size=ks)

        # scale 1/8
        self.de_conv4_1_1 = upconv(ch5, ch4)
        self.de_conv4_1_2 = GLPBlock(ch4, out_channels)

        self.de_conv4_c = conv_norelu(ch4 * 2 + self.out_channels, ch4, kernel_size=1, stride=1)
        self.de_conv4_2 = resnet_block(ch4, kernel_size=ks)
        self.de_conv4_3 = resnet_block(ch4, kernel_size=ks)
        self.de_conv4_4 = resnet_block(ch4, kernel_size=ks)

        # scale 1/4
        self.de_conv3_1_1 = upconv(ch4, ch3)
        self.de_conv3_1_2 = GLPBlock(ch3, out_channels)

        self.de_conv3_c = conv_norelu(ch3 * 2 + self.out_channels, ch3, kernel_size=1, stride=1)
        self.de_conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.de_conv3_3 = resnet_block(ch3, kernel_size=ks)
        self.de_conv3_4 = resnet_block(ch3, kernel_size=ks)

        # scale 1/2
        self.de_conv2_1_1 = upconv(ch3, ch2)
        self.de_conv2_1_2 = GLPBlock(ch2, out_channels)

        self.de_conv2_c = conv_norelu(ch2 * 2 + self.out_channels, ch2, kernel_size=1, stride=1)
        self.de_conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.de_conv2_3 = resnet_block(ch2, kernel_size=ks)
        self.de_conv2_4 = resnet_block(ch2, kernel_size=ks)

        # scale 1/1
        self.de_conv1_1_1 = upconv(ch2, ch1)
        self.de_conv1_1_2 = GLPBlock(ch1, out_channels)

        self.de_conv1_c = conv_norelu(ch1 * 2 + self.out_channels, ch1, kernel_size=1, stride=1)
        self.de_conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.de_conv1_3 = resnet_block(ch1, kernel_size=ks)
        self.de_conv1_4 = resnet_block(ch1, kernel_size=ks)

        # output layers
        self.out_convs = nn.Sequential(
            *[conv_norelu(ch_list[i], out_channels - 3 * self.num_alpha, kernel_size=ks) for i in range(6)])

    def forward(self, rgb, weights=None):

        rgb_back_all = []
        ref_all = []

        # initialize the reflectance
        b = rgb.shape[0]
        basic_vectors_transpose = self.basic_vectors_transpose.repeat(b, 1, 1)
        temp = self.temp.repeat(b, 1, 1)
        null_vectors = self.null_vectors.repeat(b, 1, 1)

        # multi-rgb fusion
        conv0_1 = torch.cat([self.en_conv0[i](rgb[:, i * 3:(i + 1) * 3, :, :])
                             for i in range(self.num_alpha)], 1)

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

        # scale 1/16
        conv5_d = self.en_conv5_1(conv4_d)
        conv5_d = self.en_conv5_4(self.en_conv5_3(self.en_conv5_2(conv5_d)))

        # scale 1/32
        conv6_d = self.en_conv6_1(conv5_d)
        conv6_d = self.en_conv6_4(self.en_conv6_3(self.en_conv6_2(conv6_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 5, None, conv6_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        # scale 1/16
        conv5_skip = self.de_conv5_1_1(conv6_d)
        ref = self.de_conv5_1_2(conv5_d, ref)
        conv5_skip = self.feature_crop(conv5_skip, conv5_d)

        conv5_d = self.de_conv5_c(torch.cat([conv5_skip, conv5_d, ref], 1))
        conv5_d = self.de_conv5_4(self.de_conv5_3(self.de_conv5_2(conv5_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 4, ref_null, conv5_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        # scale 1/8
        conv4_skip = self.de_conv4_1_1(conv5_d)
        ref = self.de_conv4_1_2(conv4_d, ref)
        conv4_skip = self.feature_crop(conv4_skip, conv4_d)

        conv4_d = self.de_conv4_c(torch.cat([conv4_skip, conv4_d, ref], 1))
        conv4_d = self.de_conv4_4(self.de_conv4_3(self.de_conv4_2(conv4_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 3, ref_null, conv4_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        # scale 1/4
        conv3_skip = self.de_conv3_1_1(conv4_d)
        ref = self.de_conv3_1_2(conv3_d, ref)
        conv3_skip = self.feature_crop(conv3_skip, conv3_d)

        conv3_d = self.de_conv3_c(torch.cat([conv3_skip, conv3_d, ref], 1))
        conv3_d = self.de_conv3_4(self.de_conv3_3(self.de_conv3_2(conv3_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 2, ref_null, conv3_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        # scale 1/2
        conv2_skip = self.de_conv2_1_1(conv3_d)
        ref = self.de_conv2_1_2(conv2_d, ref)
        conv2_skip = self.feature_crop(conv2_skip, conv2_d)

        conv2_d = self.de_conv2_c(torch.cat([conv2_skip, conv2_d, ref], 1))
        conv2_d = self.de_conv2_4(self.de_conv2_3(self.de_conv2_2(conv2_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 1, ref_null, conv2_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        # scale 1/1
        conv1_skip = self.de_conv1_1_1(conv2_d)
        ref = self.de_conv1_1_2(conv1_d, ref)
        conv1_skip = self.feature_crop(conv1_skip, conv1_d)

        conv1_d = self.de_conv1_c(torch.cat([conv1_skip, conv1_d, ref], 1))
        conv1_d = self.de_conv1_4(self.de_conv1_3(self.de_conv1_2(conv1_d)))

        ref, ref_null, rgb_back = self.ref_generation(basic_vectors_transpose, temp,
                                                      null_vectors, rgb, 0, ref_null, conv1_d)
        ref_all.insert(0, ref)
        rgb_back_all.insert(0, rgb_back)

        return ref_all, rgb_back_all

    def sub_space(self):

        illumination = self.illumination.view(1, self.num_alpha, 1, -1).repeat(1, 1, 3, 1)
        camera_sensitivity = self.camera_sensitivity.view(1, self.num_alpha, 3, -1)

        # calculate basic vectors of subspace and projection matrix
        basic_vectors_transpose = (camera_sensitivity * illumination).view(1, 3 * self.num_alpha, -1)
        basic_vectors = basic_vectors_transpose.permute(0, 2, 1)

        inverse_matrix = torch.linalg.inv(
            torch.matmul(basic_vectors_transpose.to(torch.float64), basic_vectors.to(torch.float64))).float()

        temp = torch.matmul(basic_vectors.to(torch.float64), inverse_matrix.to(torch.float64)).float()

        # print(torch.max(temp))
        # print(torch.min(temp))
        # print(torch.mean(temp))

        subspace = torch.matmul(temp.to(torch.float64), basic_vectors_transpose.to(torch.float64)).float()

        nullspace = torch.eye(self.out_channels, device=subspace.device).unsqueeze(0) - subspace

        null_vectors, _, _ = torch.linalg.svd(nullspace)
        null_vectors = null_vectors[:, :, :self.out_channels - 3 * self.num_alpha]

        return basic_vectors_transpose, temp, null_vectors

    def feature_crop(self, feature_skip, feature_d):

        if feature_skip.shape[2] > feature_d.shape[2] or feature_skip.shape[3] > feature_d.shape[3]:
            h_diff = feature_skip.shape[2] - feature_d.shape[2]
            w_diff = feature_skip.shape[3] - feature_d.shape[3]
            _, _, h, w = feature_d.shape
            feature_skip = feature_skip[:, :, h_diff // 2:h_diff // 2 + h, w_diff // 2:w_diff // 2 + w]

        return feature_skip

    def ref_generation(self, basic_vectors_transpose, temp, null_vectors, rgb, index, ref_null=None, feature=None):

        scaled_rgb = rgb

        if feature.shape[2] != scaled_rgb.shape[2] or feature.shape[3] != scaled_rgb.shape[3]:
            scaled_rgb = F.interpolate(scaled_rgb, size=(feature.shape[2], feature.shape[3]), mode='bilinear')

        ref_parallel = torch.einsum('bnc,bcij->bnij', temp.to(torch.float64),
                                    scaled_rgb.to(torch.float64)).float()

        ref_null = self.out_convs[index](feature)

        ref = ref_parallel + torch.einsum('bnc,bcij->bnij', null_vectors.to(torch.float64),
                                          ref_null.to(torch.float64)).float()

        if self.training:
            rgb_back = torch.einsum('bnc,bcij->bnij', basic_vectors_transpose.to(torch.float64),
                                    ref.to(torch.float64)).float()

            return ref, ref_null, rgb_back
        else:
            return ref, ref_null, None
