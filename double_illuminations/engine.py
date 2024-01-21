# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in train.py
"""
import math
import os
import random
import sys
import time
from typing import Iterable

import cv2
import torch
import numpy as np

import torch.nn.functional as F
from collections import OrderedDict


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, lr_scheduler,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, writer, batch_size, max_norm: float = 0):
    model.train()

    iter_num = math.ceil(len(data_loader.dataset) / batch_size)

    for iteration, (rgb_img, hyper_img, sensitivity, illumination) in enumerate(data_loader):
        rgb_img = rgb_img.to(device)
        hyper_img = hyper_img.to(device)
        sensitivity = sensitivity.to(device)
        illumination = illumination.to(device)
        # hyper_img = None

        # for j in range(white_rgb_img.shape[0]):
        #     img1 = white_rgb_img[j, :, :, :].permute(1, 2, 0).cpu().numpy()
        #     img2 = amber_rgb_img[j, :, :, :].permute(1, 2, 0).cpu().numpy()
        #
        #     cv2.imshow("img1", img1)
        #     cv2.imshow("img2", img2)
        #
        #     cv2.waitKey(0)

        # _, _, height, width = rgb_img.shape
        # rgb_list = [rgb_img]
        # for step in range(1, 7):
        #     rgb_list.append(F.interpolate(rgb_img, (height // (2 ** step), width // (2 ** step)), mode='bilinear'))

        pred_sensitivity, ref_all, rgb_back = model(rgb_img, illumination)

        primary_loss, auxiliary_loss = criterion(pred_sensitivity, ref_all, rgb_back, rgb_img, sensitivity,
                                                 hyper_img)

        loss = primary_loss + auxiliary_loss

        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        lr_scheduler.step()

        writer.add_scalar('training/primary_loss', primary_loss.item(), epoch * iter_num + iteration)
        writer.add_scalar('training/auxiliary_loss', auxiliary_loss.item(), epoch * iter_num + iteration)

        print('Epoch: [{}], iteration: [{}], loss: [{}]'.format(epoch, iteration, loss.item()))
        print()
    del loss, pred_sensitivity, ref_all, rgb_back

    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, data_loader, device, result_dir):
    model.eval()

    MRAE_list = []
    RMSE_list = []
    SAM_list = []

    for i, (rgb_list, hyper_img, sensitivity, illumination, hyper_path) in enumerate(data_loader):
        dataset_name = hyper_path[0].split('/')[-2]
        image_name = hyper_path[0].split('/')[-1]

        hyper_img = hyper_img.to(device)
        sensitivity = sensitivity.to(device)
        illumination = illumination.to(device)

        print(hyper_path[0])

        for index in range(len(rgb_list)):
            rgb_img = rgb_list[index].to(device)

            print(rgb_img.shape)

            # _, _, height, width = rgb_img.shape

            # rgb_list = [rgb_img]
            # for step in range(1, 7):
            #     rgb_list.append(F.interpolate(rgb_img, (height // (2 ** step), width // (2 ** step)), mode='bilinear'))

            pred_sensitivity, ref_all, rgb_back = model(rgb_img, illumination)

            os.makedirs(os.path.join(result_dir, dataset_name, str(index)), exist_ok=True)
            np.save(os.path.join(result_dir, dataset_name, str(index), image_name),
                    ref_all[0].squeeze().permute(1, 2, 0).cpu().numpy())

            # print("saving")

            # last_ref_max = torch.max(last_ref, 1, keepdim=True)[0]
            # last_ref_min = torch.min(last_ref, 1, keepdim=True)[0]
            # hyper_img_max = torch.max(hyper_img, 1, keepdim=True)[0]
            # hyper_img_min = torch.min(hyper_img, 1, keepdim=True)[0]

            # last_ref = (last_ref - last_ref_min) / (last_ref_max - last_ref_min)
            # hyper_img = (hyper_img - hyper_img_min) / (hyper_img_max - hyper_img_min)

            MRAE = torch.mean(torch.abs(
                ref_all[0] / torch.max(ref_all[0], 1, keepdim=True)[0] -
                hyper_img / torch.max(hyper_img, 1, keepdim=True)[0]))

            RMSE = torch.sqrt(torch.mean(torch.square(
                ref_all[0] / torch.max(ref_all[0], 1, keepdim=True)[0] -
                hyper_img / torch.max(hyper_img, 1, keepdim=True)[0])))

            SAM = torch.mean(
                torch.arccos(
                    torch.sum(ref_all[0] * hyper_img, 1, keepdim=True) / (
                            torch.sqrt(torch.sum(torch.square(ref_all[0]), 1, keepdim=True)) * torch.sqrt(
                        torch.sum(torch.square(hyper_img), 1, keepdim=True)))
                )

            )

            print("MRAE:{}, RMSE:{}, SAM:{}".format(MRAE.detach().cpu().item(), RMSE.detach().cpu().item(),
                                                    SAM.detach().cpu().item()))


            MRAE_list.append(MRAE.item())
            RMSE_list.append(RMSE.item())
            SAM_list.append(SAM.item())

    print(np.mean(np.array(MRAE_list)))
    print(np.mean(np.array(RMSE_list)))
    print(np.mean(np.array(SAM_list)))

    del pred_sensitivity, ref_all, rgb_back, MRAE, RMSE, SAM
    torch.cuda.empty_cache()

    return sum(MRAE_list) / len(MRAE_list), sum(RMSE_list) / len(RMSE_list), sum(SAM_list) / len(SAM_list)


@torch.no_grad()
def real_evaluate(model, data_loader, device, result_dir):
    model.eval()

    MRAE_list = []
    RMSE_list = []

    for i, (rgb_img, hyper_img, sensitivity, illumination, hyper_path) in enumerate(data_loader):
        image_name = hyper_path[0].split('/')[-1].split(".")[0] + ".npy"

        hyper_img = hyper_img.to(device)

        illumination = illumination.to(device)

        rgb_img = rgb_img.to(device)

        pred_sensitivity, ref_all, rgb_back = model(rgb_img, illumination)

        os.makedirs(os.path.join(result_dir, "real"), exist_ok=True)

        ref = ref_all[0].squeeze().permute(1, 2, 0).cpu().numpy()

        np.save(os.path.join(result_dir, "real", image_name), ref)

        # for i in range(hyper_img.shape[1]):
        #     gt_channel = hyper_img[0, i, :, :].cpu().numpy()
        #     ref_channel = ref[:, :, i]
        #
        #     gt_channel = gt_channel / np.max(gt_channel)
        #     ref_channel = ref_channel / np.max(ref_channel)
        #
        #     cv2.imshow("gt_channel", gt_channel)
        #     cv2.imshow("ref_channel", ref_channel)
        #
        #     cv2.waitKey(0)

        print("saving")

        MRAE = torch.mean(torch.abs(ref_all[0] / torch.max(ref_all[0]) - hyper_img / torch.max(hyper_img)))
        RMSE = torch.mean(torch.square(ref_all[0] / torch.max(ref_all[0]) - hyper_img / torch.max(hyper_img)))

        MRAE_list.append(MRAE.item())
        RMSE_list.append(RMSE.item())

    del pred_sensitivity, ref_all, rgb_back, MRAE, RMSE
    torch.cuda.empty_cache()

    return sum(MRAE_list) / len(MRAE_list), sum(RMSE_list) / len(RMSE_list)
