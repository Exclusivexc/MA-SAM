import os
import pickle
import sys
import json

sys.path.append("../MA-SAM")
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from augmentation import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from typing import *
import random


def get_aug_image(image: ndarray) -> ndarray:
    aug_dict = {}
    image = (image - image.min()) / (image.max() - image.min())
    image, gamma = gamma_correction(image)
    aug_dict["gamma"] = float(gamma)
    image, axis = random_flip(image)
    aug_dict["flip"] = float(axis)
    image, angle = random_rotate(image)
    aug_dict["rotate"] = float(angle)
    image, bits = posterization(image)
    aug_dict["posterization"] = float(bits)
    image, factor = contrast_adjustment(image)
    aug_dict["contrast"] = float(factor)
    image, factor = sharpness_enhancement(image)
    aug_dict["sharpness"] = float(factor)
    image, factor = brightness_modification(image) 
    aug_dict["brightness"] = float(factor)

    return image, aug_dict


def get_mask(image: ndarray) -> List[ndarray]:
    width, height = image.shape[:2]
    masks = mask_generator.generate(image)
    solid_masks = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
    for mask in masks:
        # print(mask["predicted_iou"], mask["stability_score"], mask['area'], width, height, width * height)
        if (
            mask["predicted_iou"] > 0.95
            and mask["stability_score"] > 0.95
            and mask["area"] < width * height / 4
        ):
            solid_masks |= mask["segmentation"]
    # print('-' * 100)
    solid_masks[solid_masks > 0] = 1
    return solid_masks


""" load model """
model_type = "vit_b"  # 根据需求选择模型类型，如 'vit_h', 'vit_b' 等
checkpoint = "/media/ubuntu/maxiaochuan/MA-SAM/sam_vit_b_01ec64.pth"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam = sam.to(device="cuda:0" if torch.cuda.is_available() else "cpu")

mask_generator = SamAutomaticMaskGenerator(sam)

""" load data """
image_dir = "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices"
masks = []

# JSON 文件路径

# 初始化日志数据
log_data = {}

files = os.listdir(image_dir)
files = [
    file
    for file in files
    if "segmentation" not in file
    and "uncertainty" not in file
    and "feature" not in file
    and "csv" not in file
]
k_times_augs = [10, 5, 15, 20]
for k_times_aug in k_times_augs:
    log_file_path = (
        f"/media/ubuntu/maxiaochuan/MA-SAM/UC-SAM/uncertainty_{k_times_aug}log.json"
    )
    for file in tqdm(files):
        # id = int(file[4:6])
        # if "segmentation" in file or "uncertainty" in file or "csv" in file or id >= 50:
        #     continue

        image_path = os.path.join(image_dir, file)
        image = pickle.load(open(image_path, "rb"))
        image = image.repeat(3, axis=2)  # 512, 512, 1 -> 512, 512, 3

        masks = []
        aug_dicts = []  # 用于存储每次增强的参数

        for _ in tqdm(range(k_times_aug)):
            aug_image, aug_dict = get_aug_image(image)
            mask = get_mask(aug_image)
            if "rotate" in aug_dict:
                mask, _ = random_rotate(mask, 360 - int(aug_dict["rotate"]))
            if "flip" in aug_dict:
                mask, _ = random_flip(mask, int(aug_dict["flip"]))

            masks.append(mask)
            aug_dicts.append(aug_dict)  # 保存当前增强的参数

        # 计算不确定性
        uncertainty = np.zeros((image.shape[0], image.shape[1]))
        for mask in masks:
            uncertainty += mask

        uncertainty /= k_times_aug
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                p = uncertainty[i][j]
                if p != 0 and p != 1:
                    uncertainty[i][j] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                else:
                    uncertainty[i][j] = 0

        # 保存不确定性文件
        uncertainty_path = image_path.replace(".pkl", f"_uncertainty_{k_times_aug}.pkl")
        pickle.dump(uncertainty, open(uncertainty_path, "wb"))

        if not os.path.exists(image_path.replace(".pkl", "_feature.pkl")):
            with torch.no_grad():
                image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
                image = (image - image.min()) / (image.max() - image.min()) * 255
                image = sam.preprocess(image)
                feature = sam.image_encoder(image)
                pickle.dump(feature.cpu().numpy(), open(image_path.replace(".pkl", "_feature.pkl"), "wb"))

        # 生成日志信息
        log_message = {
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_max": float(uncertainty.max()),
            "uncertainty_min": float(uncertainty.min()),
            "aug_dicts": aug_dicts,  # 保存增强参数
        }

        # 将日志信息添加到 JSON 数据中
        log_data[file] = log_message
        tqdm.write(
            f"Processed {file}, uncertainty_mean: {uncertainty.mean()}, uncertainty_std:{uncertainty.std()}, uncertainty_max: {uncertainty.max()}"
        )

    # 保存日志数据到 JSON 文件
    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)

    print(f"Log data has been saved to {log_file_path}")
