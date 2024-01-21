import glob

import numpy as np

import os

image_path = "/media/dong/c62b488e-fe48-41ce-beaf-cdb01f81e1d6/spectral_dataset/training"
target_path = "/media/dong/c62b488e-fe48-41ce-beaf-cdb01f81e1d6/spectral_dataset/training_patches128"

os.makedirs(target_path, exist_ok=True)


image_list = glob.glob(os.path.join(image_path, "*", "*.npy"))

patch_size = 128
step_size = 96

patch_index = 0

for image_path in image_list:
    print(image_path)
    img = np.load(image_path)

    h, w, _ = img.shape

    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            if i + patch_size <= h and j + patch_size <= w:
                patch = img[i:i + patch_size, j:j + patch_size, :]
                np.save(os.path.join(target_path, str(patch_index) + ".npy"), patch)
            elif i + patch_size > h and j + patch_size <= w:
                patch = img[h - patch_size:h, j:j + patch_size, :]
                np.save(os.path.join(target_path, str(patch_index) + ".npy"), patch)
            elif i + patch_size <= h and j + patch_size > w:
                patch = img[i:i + patch_size, w - patch_size:w, :]
                np.save(os.path.join(target_path, str(patch_index) + ".npy"), patch)
            else:
                patch = img[h - patch_size:h, w - patch_size:w, :]
                np.save(os.path.join(target_path, str(patch_index) + ".npy"), patch)

            patch_index += 1

    print(patch_index)