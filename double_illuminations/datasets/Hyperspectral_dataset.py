# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torch.utils.data as data
import torch
import torch.nn.functional as F

from datasets import load_sensitivity
from datasets import load_illumination

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from scipy.io import loadmat

class Training_dataset(data.Dataset):
    def __init__(self, img_folder):

        self.img_folder = img_folder

        self.image_path_list = []

        self.image_path_list = glob.glob(os.path.join(img_folder, "*.npy"))

        self.illumination_1 = np.load("datasets/white_illumination.npy")[2:]
        self.illumination_2 = np.load("datasets/amber_30_illumination.npy")[2:]

        peak_value = max(np.max(self.illumination_1), np.max(self.illumination_2))

        # keep the relative scale of two illumination
        self.illumination_1 = self.illumination_1 / peak_value
        self.illumination_2 = self.illumination_2 / peak_value

        self.camera_shuffle_index = np.load("datasets/response_shuffle_index.npy")[:-5]

        self.sensitivity_list = np.array(load_sensitivity.load())[self.camera_shuffle_index, :, 2:]

        self.illumination = np.concatenate([self.illumination_1.reshape(1, -1), self.illumination_2.reshape(1, -1)], 0)

    def random_flip(self, hyper_img):

        if np.random.uniform(0, 1) < 0.5:
            hyper_img = np.flip(hyper_img, 0).copy()

        if np.random.uniform(0, 1) < 0.5:
            hyper_img = np.flip(hyper_img, 1).copy()

        return hyper_img

    def random_scale(self, white_rgb_img):

        scale = 1.0
        if np.random.uniform(0, 1) < 0.9:
            scale = np.power(10, np.random.uniform(-1, 1))
            white_rgb_img = white_rgb_img * scale

        return white_rgb_img, scale

    def sensitivity_generation(self):

        # fig, ax_list = plt.subplots(2)

        num = np.random.randint(1, 3)

        index_list = np.arange(0, self.sensitivity_list.shape[0])
        np.random.shuffle(index_list)
        selected_index = index_list[:num]

        sensitivity = self.sensitivity_list[selected_index, :, :]
        weights = np.random.random(num)
        sensitivity = np.einsum("n,nij->ij", weights, sensitivity)

        # ax_list[0].plot(range(420, 730, 10), sensitivity[2], 'b-', range(420, 730, 10), sensitivity[1], 'g-',
        #                 range(420, 730, 10), sensitivity[0], 'r-')

        sensitivity = self.random_sens_noise(sensitivity)

        sensitivity = sensitivity / np.max(sensitivity)

        # ax_list[1].plot(range(420, 730, 10), sensitivity[2], 'b-', range(420, 730, 10), sensitivity[1], 'g-',
        #                 range(420, 730, 10), sensitivity[0], 'r-')
        # plt.show()

        return sensitivity

    def random_sens_noise(self, sensitivity):

        if np.random.uniform(0, 1) < 0.5:
            sensitivity = sensitivity + (np.random.random(sensitivity.shape) - 0.5) * 0.1

        return sensitivity

    def interpolate(self, hyper_img, sensitivity):

        h, w, c = hyper_img.shape
        hyper_img = hyper_img.reshape((-1, c))
        hyper_f = interp1d(range(420, 730, 10), hyper_img, "slinear")
        inter_hyper_img = hyper_f(range(420, 721, 1))

        inter_hyper_img = inter_hyper_img.reshape((h, w, -1))

        f = interp1d(range(420, 730, 10), sensitivity, "slinear")
        inter_sensitivity = f(range(420, 721, 1))

        f = interp1d(range(420, 730, 10), self.illumination_1, "slinear")
        inter_illumination_1 = f(range(420, 721, 1))

        f = interp1d(range(420, 730, 10), self.illumination_2, "slinear")
        inter_illumination_2 = f(range(420, 721, 1))

        inter_rgb_img_1 = np.einsum("ijk,nk->ijn", inter_hyper_img, inter_sensitivity * inter_illumination_1) / 41.4
        inter_rgb_img_2 = np.einsum("ijk,nk->ijn", inter_hyper_img, inter_sensitivity * inter_illumination_2) / 41.4

        return inter_rgb_img_1, inter_rgb_img_2

    def __getitem__(self, idx):

        hyper_path = self.image_path_list[idx]
        hyper_img = np.load(hyper_path)
        hyper_img = self.random_flip(hyper_img)

        sensitivity = self.sensitivity_generation()

        # 4.14 is the largest value of the rgb
        # rgb_img_1 = np.einsum("ijk,nk->ijn", hyper_img, sensitivity * self.illumination_1) / 4.14
        # rgb_img_2 = np.einsum("ijk,nk->ijn", hyper_img, sensitivity * self.illumination_2) / 4.14

        rgb_img_1, rgb_img_2 = self.interpolate(hyper_img, sensitivity)

        # 2.6 is the largest value of the hyper
        hyper_img = hyper_img / 2.6

        # cv2.imshow("rgb_img_1", rgb_img_1)
        # cv2.imshow("rgb_img_2", rgb_img_2)
        # cv2.waitKey(0)

        rgb_img_stack = np.concatenate([rgb_img_1, rgb_img_2], -1)

        return ToTensor(rgb_img_stack), ToTensor(hyper_img), sensitivity, self.illumination

    def __len__(self):
        return len(self.image_path_list)

    # def random_crop(self, hyper_img):
    #
    #     h, w, _ = hyper_img.shape
    #     max_h = h - self.patch_size
    #     max_w = w - self.patch_size
    #
    #     if max_h == 0:
    #         start_h = 0
    #     else:
    #         start_h = np.random.randint(0, max_h)
    #
    #     if max_w == 0:
    #         start_w = 0
    #     else:
    #         start_w = np.random.randint(0, max_w)
    #
    #     return hyper_img[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size, :]


class Testing_dataset(data.Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder

        self.image_path_list = []

        self.image_path_list = glob.glob(os.path.join(img_folder, "*", "*.npy"))

        self.illumination_1 = np.load("datasets/white_illumination.npy")[2:]
        self.illumination_2 = np.load("datasets/amber_30_illumination.npy")[2:]

        peak_value = max(np.max(self.illumination_1), np.max(self.illumination_2))

        # keep the relative scale of two illumination
        self.illumination_1 = self.illumination_1 / peak_value
        self.illumination_2 = self.illumination_2 / peak_value

        self.camera_shuffle_index = np.load("datasets/response_shuffle_index.npy")[-5:]

        self.sensitivity_list = np.array(load_sensitivity.load())[self.camera_shuffle_index, :, 2:]

        self.illumination = np.concatenate([self.illumination_1.reshape(1, -1), self.illumination_2.reshape(1, -1)], 0)

    def interpolate(self, hyper_img, sensitivity):
        h, w, c = hyper_img.shape
        hyper_img = hyper_img.reshape((-1, c))
        hyper_f = interp1d(range(420, 730, 10), hyper_img, "slinear", bounds_error=True)
        inter_hyper_img = hyper_f(range(420, 721, 1))

        inter_hyper_img = inter_hyper_img.reshape((h, w, -1))

        c, n, k = sensitivity.shape
        sensitivity = sensitivity.reshape((-1, k))
        f = interp1d(range(420, 730, 10), sensitivity, "slinear")
        inter_sensitivity = f(range(420, 721, 1))
        inter_sensitivity = inter_sensitivity.reshape((c, n, -1))

        f = interp1d(range(420, 730, 10), self.illumination_1, "slinear")
        inter_illumination_1 = f(range(420, 721, 1))

        f = interp1d(range(420, 730, 10), self.illumination_2, "slinear")
        inter_illumination_2 = f(range(420, 721, 1))

        inter_rgb_img_1 = np.einsum("ijk,cnk->cijn", inter_hyper_img, inter_sensitivity * inter_illumination_1) / 41.4
        inter_rgb_img_2 = np.einsum("ijk,cnk->cijn", inter_hyper_img, inter_sensitivity * inter_illumination_2) / 41.4

        return inter_rgb_img_1, inter_rgb_img_2

    def __getitem__(self, idx):
        hyper_path = self.image_path_list[idx]
        hyper_img = np.load(hyper_path)

        sensitivity = self.sensitivity_list

        sensitivity = sensitivity.reshape((-1, 3 * 31))

        sensitivity = sensitivity / np.max(sensitivity, -1).reshape((-1, 1))

        sensitivity = sensitivity.reshape((-1, 3, 31))

        # 4.14 is the largest value of the rgb
        # rgb_img_1 = np.einsum("ijk,nk->ijn", hyper_img, sensitivity * self.illumination_1) / 4.14
        # rgb_img_2 = np.einsum("ijk,nk->ijn", hyper_img, sensitivity * self.illumination_2) / 4.14

        rgb_img_1, rgb_img_2 = self.interpolate(hyper_img, sensitivity)

        # 2.6 is the largest value of the hyper
        hyper_img = hyper_img / 2.6

        # cv2.imshow("rgb_img_1", rgb_img_1)
        # cv2.imshow("rgb_img_2", rgb_img_2)
        # cv2.waitKey(0)

        rgb_img_stack = np.concatenate([rgb_img_1, rgb_img_2], -1)

        rgb_list = [ToTensor(rgb_img_stack[i]) for i in range(rgb_img_stack.shape[0])]

        return rgb_list, ToTensor(hyper_img), sensitivity, self.illumination, hyper_path

    def __len__(self):
        return len(self.image_path_list)

class Real_dataset(data.Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder

        self.image_path_list = []

        self.image_path_list = glob.glob(os.path.join(img_folder, "*.mat"))

        self.illumination_1 = np.load("datasets/white_illumination.npy")[2:]
        self.illumination_2 = np.load("datasets/amber_30_illumination.npy")[2:]

        peak_value = max(np.max(self.illumination_1), np.max(self.illumination_2))

        # keep the relative scale of two illumination
        self.illumination_1 = self.illumination_1 / peak_value
        self.illumination_2 = self.illumination_2 / peak_value

        self.illumination = np.concatenate([self.illumination_1.reshape(1, -1), self.illumination_2.reshape(1, -1)], 0)

    def __getitem__(self, idx):
        hyper_path = self.image_path_list[idx]
        mat_file = loadmat(hyper_path)

        rgb_white = mat_file["raw_white"]
        rgb_amber = mat_file["raw_amber"]

        # rgb_white = cv2.GaussianBlur(rgb_white, (5, 5), 0.7)
        # rgb_amber = cv2.GaussianBlur(rgb_amber, (5, 5), 0.7)
        rgb = np.concatenate([rgb_white, rgb_amber], -1)

        ref = mat_file["reflectance"][:, :, 2:]

        sensitivity = 0

        return ToTensor(rgb), ToTensor(ref), sensitivity, self.illumination, hyper_path

    def __len__(self):
        return len(self.image_path_list)


def ToTensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float()
