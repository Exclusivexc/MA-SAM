import json
import pickle
import numpy as np
from augmentation import *
import random
from numpy import ndarray

def get_aug_image(image: ndarray) -> ndarray:
    aug_dict = {}
    image = (image - image.min()) / (image.max() - image.min())
    image, gamma = gamma_correction(image)
    aug_dict["gamma"] = float(gamma)
    if random.random() > 0.5:
        image, axis = random_flip(image)
        aug_dict["flip"] = float(axis)
    if random.random() > 0.5:
        image, angle = random_rotate(image)
        aug_dict["rotate"] = float(angle)
    if random.random() > 0.5:
        image, bits = posterization(image)
        aug_dict["posterization"] = float(bits)
    if random.random() > 0.5:
        image, factor = contrast_adjustment(image)
        aug_dict["contrast"] = float(factor)
    if random.random() > 0.5:
        image, factor = sharpness_enhancement(image)
        aug_dict["sharpness"] = float(factor)

    return image, aug_dict


image = pickle.load(open("/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices/Case00_slice00.pkl", "rb"))
image = image.repeat(3, axis=2)
print(image.shape)
aug_dicts = []  # 用于存储每次增强的参数
for _ in range(10):
    aug_image, aug_dict = get_aug_image(image)
    aug_dicts.append(aug_dict)
    # aug_dicts = {"rotate": 90, "flip": 1}
uncertainty = np.array([1, 2, 3])
file = "acdc"
log_data = {}
log_message = {
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_max": float(uncertainty.max()),
            "uncertainty_min": float(uncertainty.min()),
            "aug_dicts": aug_dicts,  # 保存增强参数
        }

        # 将日志信息添加到 JSON 数据中
log_data[file] = log_message
log_file_path = "/media/ubuntu/maxiaochuan/MA-SAM/UC-SAM/log.json"
with open(log_file_path, "w") as log_file:
    json.dump(log_data, log_file, indent=4)