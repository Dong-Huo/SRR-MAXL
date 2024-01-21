#!/usr/local/bin/python
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class reconstruct_loss(nn.Module):
    """the loss between the input and synthesized input"""

    def __init__(self):
        super(reconstruct_loss, self).__init__()

    def forward(self, pred_sensitivity, ref_all, rgb_back, rgb_img, sensitivity=None, hyper_img=None):
        rgb_back_loss = []

        # scales = torch.ones_like(scales)

        # rgb = rgb / scales
        # hyper_img = hyper_img / scales

        primary_loss = 0

        # for i in range(len(rgb_back_all)):
        #
        #     if rgb_back_all[i].shape[2] != rgb_img.shape[2]:
        #         scaled_rgb_img = F.interpolate(rgb_img, size=(rgb_back_all[i].shape[2], rgb_back_all[i].shape[3]),
        #                                        mode='bilinear')
        #         rgb_back_loss.append(torch.mean(torch.abs(rgb_back_all[i] - scaled_rgb_img)))
        #     else:
        #         rgb_back_loss.append(torch.mean(torch.abs(rgb_back_all[i] - rgb_img)))
        #
        # rgb_back_loss = sum(rgb_back_loss)
        # null_back_loss = sum(null_back_loss)
        # delta_null_back_loss = sum(delta_null_back_loss)

        # for i in range(rgb_back_all[0].shape[0]):
        #     gt = rgb[i, :, :, :].permute(1, 2, 0).cpu().numpy()
        #     est = rgb_back_all[0][i, :, :, :].permute(1, 2, 0).cpu().numpy()
        #
        #     cv2.imshow("gt", gt)
        #     cv2.imshow("est", est)
        #     cv2.waitKey(0)

        # print("rgb_back_loss:{}".format(rgb_back_loss.item()))

        # primary_loss += rgb_back_loss

        ref_loss_list = []

        if hyper_img is not None:
            for i in range(len(ref_all)):

                if ref_all[i].shape[2] != hyper_img.shape[2]:
                    scaled_hyper_img = F.interpolate(hyper_img, size=(ref_all[i].shape[2], ref_all[i].shape[3]),
                                                     mode='bilinear')
                    ref_loss_list.append(torch.mean(torch.abs(ref_all[i] - scaled_hyper_img)))
                else:
                    ref_loss_list.append(torch.mean(torch.abs(ref_all[i] - hyper_img)))

            print("ref_loss:{}".format(str([round(ref_loss.item(), 10) for ref_loss in ref_loss_list])))

            ref_loss = sum(ref_loss_list)

            primary_loss += ref_loss

        if sensitivity is not None:
            pred_sensitivity = pred_sensitivity.contiguous().view(-1, 3 * 31)
            sensitivity = sensitivity.view(-1, 3 * 31)

            pred_sensitivity = pred_sensitivity / torch.max(pred_sensitivity, -1)[0].view(-1, 1)
            sensitivity = sensitivity / torch.max(sensitivity, -1)[0].view(-1, 1)

            sens_loss = torch.mean(torch.abs(pred_sensitivity - sensitivity))
            primary_loss += sens_loss

            print("sens_loss:{}".format(str(round(sens_loss.item(), 10))))

            # pred_de = pred_sensitivity[:, 1:, :] - pred_sensitivity[:, :-1, :]
            # de = sensitivity[:, 1:, :] - sensitivity[:, :-1, :]
            #
            # de_sens_loss = torch.mean(
            #     torch.square(pred_de / torch.max(pred_de) - de / torch.max(de)))
            # primary_loss += de_sens_loss
            #
            # print("de_sens_loss:{}".format(str(round(de_sens_loss.item(), 10))))

        rgb_loss = torch.mean(torch.abs(rgb_back - rgb_img))
        auxiliary_loss = rgb_loss

        print("rgb_loss:{}".format(str(round(rgb_loss.item(), 10))))

        return primary_loss, auxiliary_loss

class meta_reconstruct_loss(nn.Module):
    """the loss between the input and synthesized input"""

    def __init__(self):
        super(meta_reconstruct_loss, self).__init__()

    def forward(self, rgb_back, rgb_img):
        rgb_loss = torch.mean(torch.abs(rgb_back - rgb_img))
        auxiliary_loss = rgb_loss

        print("rgb_loss:{}".format(str(round(rgb_loss.item(), 10))))

        return auxiliary_loss