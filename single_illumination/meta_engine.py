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

import copy

from fvcore.nn import FlopCountAnalysis


def meta_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable,
                         primary_optimizer: torch.optim.Optimizer, auxiliary_optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, writer, batch_size, num_gradient, max_norm: float = 0):
    model.train()

    iter_num = math.ceil(len(data_loader.dataset) / batch_size)

    for iteration, (rgb_img, hyper_img, sensitivity, illumination) in enumerate(data_loader):
        rgb_img = rgb_img.to(device)
        hyper_img = hyper_img.to(device)
        sensitivity = sensitivity.to(device)
        illumination = illumination.to(device)

        initial_state = copy.deepcopy(model.state_dict())

        pyramid_net_grads = None
        sensitivity_net_grads = None

        print("--------------------------------------Gradient adaptation--------------------------------------")

        primary_optimizer.param_groups[0]['lr'] = auxiliary_optimizer.param_groups[0]['lr']

        for i in range(batch_size):

            state_copy = copy.deepcopy(initial_state)

            model.load_state_dict(state_copy)

            # update theta_tilde
            for j in range(num_gradient):
                pred_sensitivity, ref_all, rgb_back = model(rgb_img[i:i + 1], illumination[i:i + 1])

                primary_loss, auxiliary_loss = criterion(pred_sensitivity, ref_all, rgb_back,
                                                         rgb_img[i:i + 1], sensitivity[i:i + 1], hyper_img[i:i + 1])

                primary_optimizer.zero_grad()
                auxiliary_optimizer.zero_grad()

                auxiliary_loss.backward()

                primary_optimizer.step()
                auxiliary_optimizer.step()


            # get gradient under theta_tilde
            pred_sensitivity, ref_all, rgb_back = model(rgb_img[i:i + 1], illumination[i:i + 1])
            primary_loss, auxiliary_loss = criterion(pred_sensitivity, ref_all, rgb_back,
                                                     rgb_img[i:i + 1], sensitivity[i:i + 1], hyper_img[i:i + 1])

            if pyramid_net_grads is None or sensitivity_net_grads is None:
                net_grads = torch.autograd.grad(primary_loss,
                                                list(model.pyramid_net.parameters()) + list(
                                                    model.sensitivity_net.parameters()))
                # sensitivity_net_grads = torch.autograd.grad(primary_loss, model.sensitivity_net.parameters(),
                #                                             create_graph=True)

                pyramid_net_grads = [x / batch_size for x in net_grads[:204]]
                sensitivity_net_grads = [x / batch_size for x in net_grads[204:]]

            else:
                new_net_grads = torch.autograd.grad(primary_loss,
                                                    list(model.pyramid_net.parameters()) + list(
                                                        model.sensitivity_net.parameters()))

                # new_sensitivity_net_grads = torch.autograd.grad(primary_loss, model.sensitivity_net.parameters(),
                #                                                 create_graph=True)

                pyramid_net_grads = [x + y / batch_size for (x, y) in zip(pyramid_net_grads, new_net_grads[:204])]
                sensitivity_net_grads = [x + y / batch_size for (x, y) in
                                         zip(sensitivity_net_grads, new_net_grads[204:])]


        state_copy = copy.deepcopy(initial_state)
        model.load_state_dict(state_copy)

        print("--------------------------------------Auxiliary update--------------------------------------")

        for i in range(batch_size):

            pred_sensitivity = model.sensitivity_net(rgb_img[i:i + 1])
            ref_all, shared_feature = model.pyramid_net(rgb_img[i:i + 1], illumination[i:i + 1], pred_sensitivity)

            # update theta_auxiliary
            for j in range(num_gradient):
                rgb_back = model.auxiliary_net(shared_feature.detach(), ref_all[0].detach())

                primary_loss, auxiliary_loss = criterion(pred_sensitivity, ref_all, rgb_back,
                                                         rgb_img[i:i + 1], sensitivity[i:i + 1], hyper_img[i:i + 1])

                auxiliary_optimizer.zero_grad()

                auxiliary_loss.backward()

                auxiliary_optimizer.step()
        
        primary_optimizer.param_groups[0]['lr'] = auxiliary_optimizer.param_groups[0]['lr'] / 200

        # update theta_primary
        primary_optimizer.zero_grad()

        for param, grad in zip(model.pyramid_net.parameters(), pyramid_net_grads):
            param.grad = grad

        for param, grad in zip(model.sensitivity_net.parameters(), sensitivity_net_grads):
            param.grad = grad

        primary_optimizer.step()

        print('Epoch: [{}], iteration: [{}]]'.format(epoch, iteration))
        print()

        del rgb_img, hyper_img, sensitivity, illumination, shared_feature
        torch.cuda.empty_cache()
        del primary_loss, auxiliary_loss, pred_sensitivity, ref_all, rgb_back
        torch.cuda.empty_cache()
        del state_copy, initial_state, pyramid_net_grads, sensitivity_net_grads, net_grads, new_net_grads
        torch.cuda.empty_cache()


def meta_evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
                  data_loader: Iterable, primary_optimizer: torch.optim.Optimizer,
                  auxiliary_optimizer: torch.optim.Optimizer,
                  device: torch.device, num_gradient, result_dir, max_norm: float = 0):
    MRAE_list = []
    RMSE_list = []
    SAM_list = []

    torch.cuda.empty_cache()

    initial_state = copy.deepcopy(model.state_dict())

    primary_optimizer.param_groups[0]['lr'] = auxiliary_optimizer.param_groups[0]['lr']

    for i, (rgb_list, hyper_img, sensitivity, illumination, hyper_path) in enumerate(data_loader):
        dataset_name = hyper_path[0].split('/')[-2]
        image_name = hyper_path[0].split('/')[-1]

        hyper_img = hyper_img.to(device)
        sensitivity = sensitivity.to(device)
        illumination = illumination.to(device)

        print(hyper_path[0])

        for index in range(len(rgb_list)):
            rgb_img = rgb_list[index].to(device)

            print("--------------------------------------Meta-testing--------------------------------------")

            state_copy = copy.deepcopy(initial_state)
            model.load_state_dict(state_copy)

            start = time.time()


            for j in range(num_gradient):
                pred_sensitivity, ref_all, rgb_back = model(rgb_img, illumination)

                auxiliary_loss = criterion(rgb_back, rgb_img)

                primary_optimizer.zero_grad()
                auxiliary_optimizer.zero_grad()

                auxiliary_loss.backward()

                primary_optimizer.step()
                auxiliary_optimizer.step()

            print(time.time() - start, " second")

            os.makedirs(os.path.join(result_dir, dataset_name, str(index)), exist_ok=True)
            # np.save(os.path.join(result_dir, dataset_name, str(index), image_name),
            #         ref_all[0].detach().squeeze().permute(1, 2, 0).cpu().numpy())

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

            del pred_sensitivity, ref_all, rgb_back, MRAE, RMSE, SAM, state_copy, auxiliary_loss
            torch.cuda.empty_cache()

    model.load_state_dict(initial_state)
    del initial_state
    torch.cuda.empty_cache()

    return sum(MRAE_list) / len(MRAE_list), sum(RMSE_list) / len(RMSE_list), sum(SAM_list) / len(SAM_list)


def meta_real_evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
                       data_loader: Iterable, primary_optimizer: torch.optim.Optimizer,
                       auxiliary_optimizer: torch.optim.Optimizer,
                       device: torch.device, num_gradient, result_dir, max_norm: float = 0):
    model.eval()

    MRAE_list = []
    RMSE_list = []

    initial_state = copy.deepcopy(model.state_dict())

    primary_optimizer.param_groups[0]['lr'] = auxiliary_optimizer.param_groups[0]['lr']

    for i, (rgb_img, hyper_img, sensitivity, illumination, hyper_path) in enumerate(data_loader):
        image_name = hyper_path[0].split('/')[-1].split(".")[0] + ".npy"

        print(hyper_path[0])

        hyper_img = hyper_img.to(device)

        illumination = illumination.to(device)

        rgb_img = rgb_img.to(device)

        state_copy = copy.deepcopy(initial_state)
        model.load_state_dict(state_copy)


        for j in range(num_gradient):
            pred_sensitivity, ref_all, rgb_back = model(rgb_img, illumination)

            auxiliary_loss = criterion(rgb_back, rgb_img)

            primary_optimizer.zero_grad()
            auxiliary_optimizer.zero_grad()

            auxiliary_loss.backward()

            primary_optimizer.step()
            auxiliary_optimizer.step()



        os.makedirs(result_dir, exist_ok=True)

        ref = ref_all[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()

        # np.save(os.path.join(result_dir, image_name), ref)

        print("saving")

        MRAE = torch.mean(torch.abs(ref_all[0] / torch.max(ref_all[0]) - hyper_img / torch.max(hyper_img)))
        RMSE = torch.mean(torch.square(ref_all[0] / torch.max(ref_all[0]) - hyper_img / torch.max(hyper_img)))

        MRAE_list.append(MRAE.item())
        RMSE_list.append(RMSE.item())
    model.load_state_dict(initial_state)
    del pred_sensitivity, ref_all, rgb_back, MRAE, RMSE
    torch.cuda.empty_cache()

