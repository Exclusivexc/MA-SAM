import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from torch.nn import functional as F
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import json

base_dir = "/media/ubuntu/maxiaochuan/MA-SAM/data"


def organize_data():
    save_pth = base_dir + "/" + "Promise12_preprocessed"

    os.makedirs(save_pth + "/imagesTr", exist_ok=True)
    os.makedirs(save_pth + "/labelsTr", exist_ok=True)

    data_pth = base_dir + "/" + "Promise12_reorganized"
    data_fd_list = os.listdir(data_pth)
    data_fd_list = [
        data_fd
        for data_fd in data_fd_list
        if data_fd.endswith(".nii.gz") and "segmentation" not in data_fd
    ]
    data_fd_list.sort()

    for data_fd in data_fd_list:
        patient_ID = "00" + data_fd[4:6]

        mask_obj = nib.load(
            base_dir
            + "/"
            + "Promise12_reorganized/"
            + data_fd.replace(".nii.gz", "_segmentation.nii.gz")
        )
        mask_arr = mask_obj.get_fdata()
        mask_arr[mask_arr > 1] = 1
        new_mask_obj = nib.Nifti1Image(
            mask_arr, mask_obj.affine, header=mask_obj.header
        )
        nib.save(new_mask_obj, save_pth + "/labelsTr/" + patient_ID + ".nii.gz")

        shutil.copy(
            base_dir + "/" + "Promise12_reorganized/" + data_fd,
            save_pth + "/imagesTr/" + patient_ID + "_0000.nii.gz",
        )


def get_3D_2D_all_5slice():
    save_pth = base_dir + "/" "Promise12/2D_all_5slice"
    print(save_pth)
    os.makedirs(save_pth, exist_ok=True)
    data_pth_all = [
        base_dir + "/Promise12_reorganized",
    ]

    for data_pth in tqdm(data_pth_all):
        data_fd_list = os.listdir(data_pth)
        data_fd_list = [
            data_fd
            for data_fd in data_fd_list
            if data_fd.endswith(".nii.gz") and "segmentation" not in data_fd
        ]
        data_fd_list.sort()

        cnt = 0
        for data_fd_indx, data_fd in enumerate(data_fd_list):
            case_id = data_fd[4:6]

            # if not os.path.exists(save_pth + "/" + case_id):
            #     os.makedirs(save_pth + "/" + case_id)
            #     os.mkdir(save_pth + "/" + case_id + "/images")
            #     os.mkdir(save_pth + "/" + case_id + "/masks")

            img_obj = nib.load(data_pth + "/" + data_fd)
            img_arr = img_obj.get_fdata()

            mask_obj = nib.load(
                data_pth + "/" + data_fd.replace(".nii.gz", "_segmentation.nii.gz")
            )
            mask_arr = mask_obj.get_fdata()

            img_arr = np.float32(img_arr)
            mask_arr = np.float32(mask_arr)

            high = np.quantile(img_arr, 0.99)
            low = np.min(img_arr)
            img_arr = np.where(img_arr > high, high, img_arr)
            lungwin = np.array([low * 1.0, high * 1.0])
            img_arr = (img_arr - lungwin[0]) / (lungwin[1] - lungwin[0])

            h, w = img_arr.shape[0], img_arr.shape[1]
            out_h, out_w = 256, 256

            if h != 256 or w != 256:
                img_arr = zoom(img_arr, (out_h / h, out_w / w, 1.0), order=3)
                mask_arr = zoom(mask_arr, (out_h / h, out_w / w, 1.0), order=0)

            img_arr = np.concatenate(
                (
                    img_arr[:, :, 0:1],
                    img_arr[:, :, 0:1],
                    img_arr,
                    img_arr[:, :, -1:],
                    img_arr[:, :, -1:],
                ),
                axis=-1,
            )
            mask_arr = np.concatenate(
                (
                    mask_arr[:, :, 0:1],
                    mask_arr[:, :, 0:1],
                    mask_arr,
                    mask_arr[:, :, -1:],
                    mask_arr[:, :, -1:],
                ),
                axis=-1,
            )

            for slice_indx in tqdm(range(2, img_arr.shape[2] - 2)):

                slice_arr = img_arr[:, :, slice_indx - 2 : slice_indx + 3]
                slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

                mask_arr_2D = mask_arr[:, :, slice_indx - 2 : slice_indx + 3]
                mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

                with open(
                    save_pth
                    + f"/Case{int(case_id):02d}_slice{(slice_indx - 2):02d}.pkl",
                    "wb",
                ) as file:
                    pickle.dump(slice_arr, file)

                with open(
                    save_pth
                    + f"/Case{int(case_id):02d}_slice{(slice_indx - 2):02d}_segmentation.pkl",
                    "wb",
                ) as file:
                    pickle.dump(mask_arr_2D, file)

            cnt += 1


def get_csv():

    save_pth = base_dir + "/prostateD/2D_all_5slice"

    training_csv = save_pth + "/training.csv"
    validation_csv = save_pth + "/validation.csv"
    test_csv = save_pth + "/test.csv"
    all_csv = save_pth + "/all.csv"

    data_fd_list = os.listdir(save_pth)
    data_fd_list = [
        data_fd
        for data_fd in data_fd_list
        if data_fd.startswith("00") and "." not in data_fd
    ]

    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)

    test_fd_list = ["0001", "0032", "0034"]

    training_fd_list = list(set(data_fd_list) - set(test_fd_list))
    validation_fd_list = random.sample(test_fd_list, min(len(test_fd_list), 4))

    path_list_all = []
    for data_fd in data_fd_list:
        slice_list = os.listdir(save_pth + "/" + data_fd + "/images")
        slice_pth_list = [data_fd + "/images/" + slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    df = pd.DataFrame(path_list_all, columns=["image_pth"])
    df["mask_pth"] = path_list_all
    df["mask_pth"] = df["mask_pth"].apply(
        lambda x: x.replace("/images/2Dimage_", "/masks/2Dmask_")
    )

    df.to_csv(all_csv, index=False)

    path_list_all = []
    for data_fd in training_fd_list:
        slice_list = os.listdir(save_pth + "/" + data_fd + "/images")
        slice_pth_list = [data_fd + "/images/" + slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    df = pd.DataFrame(path_list_all, columns=["image_pth"])
    df["mask_pth"] = path_list_all
    df["mask_pth"] = df["mask_pth"].apply(
        lambda x: x.replace("/images/2Dimage_", "/masks/2Dmask_")
    )

    df.to_csv(training_csv, index=False)

    path_list_all = []
    for data_fd in validation_fd_list:
        slice_list = os.listdir(save_pth + "/" + data_fd + "/images")
        slice_pth_list = [data_fd + "/images/" + slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    df = pd.DataFrame(path_list_all, columns=["image_pth"])
    df["mask_pth"] = path_list_all
    df["mask_pth"] = df["mask_pth"].apply(
        lambda x: x.replace("/images/2Dimage_", "/masks/2Dmask_")
    )

    df.to_csv(validation_csv, index=False)

    path_list_all = []
    for data_fd in test_fd_list:
        slice_list = os.listdir(save_pth + "/" + data_fd + "/images")
        slice_list.sort()
        slice_pth_list = [data_fd + "/images/" + slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    df = pd.DataFrame(path_list_all, columns=["image_pth"])
    df["mask_pth"] = path_list_all
    df["mask_pth"] = df["mask_pth"].apply(
        lambda x: x.replace("/images/2Dimage_", "/masks/2Dmask_")
    )

    df.to_csv(test_csv, index=False)


if __name__ == "__main__":
    # organize_data()
    get_3D_2D_all_5slice()
    # get_csv()
