import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry
from collections import defaultdict

from icecream import ic
import pandas as pd
import pickle
from datetime import datetime
from einops import repeat
from scipy.ndimage import zoom
from utils import calculate_metric_percase
import nibabel as nib
from datasets.dataset import average_pooling

HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372


def test_single_volume(
    image,
    label,
    uncertainty,
    net,
    classes,
    multimask_output,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
):

    image, label, uncertainty = image.squeeze(0), label.squeeze(0), uncertainty.squeeze(0)  # [b, h, w, d], [b, h, w, d]
    label = label[:, :, :, 0]
    # print(image.shape, label.shape) # 23, 320, 320, 1

    probability = np.expand_dims(
        np.zeros_like(label, dtype=np.float32), axis=-1
    )  # [b, h, w, c]
    probability = repeat(probability, "d h w c -> d h w (repeat c)", repeat=classes + 1)
    # print(probability.shape) # 23, 320, 320, 3


    # avg_cnt = np.ones_like(probability, dtype=np.float32)
    for ind in tqdm(range(image.shape[0])):
        slice = image[ind]
        uncertainty_slice = torch.from_numpy(uncertainty[ind]).unsqueeze(0).unsqueeze(0).cuda()
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y, 1), order=3)

        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, "b c h w d -> b (repeat c) h w d", repeat=3)
        inputs = torch.permute(inputs, (0, -1, 1, 2, 3))
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0], uncertainty_slice)
            output_masks = outputs["masks"]

            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
            out = out.cpu().detach().numpy()
            out_pred = torch.softmax(output_masks, dim=1)
            out_pred = torch.permute(out_pred, (0, 2, 3, 1))
            out_pred = out_pred.cpu().detach().numpy()
            out_h, out_w = out.shape[1], out.shape[2]
            # if x != out_h or y != out_w:
            #     out_pred = zoom(out_pred, (1.0, x / out_h, y / out_w, 1.0), order=3)


            probability[ind] += out_pred.squeeze()
            # avg_cnt[ind] += 1.0
    
    x, y = label.shape[1], label.shape[2]
    if x != patch_size[0] or y != patch_size[1]:
        label = zoom(label, (1, patch_size[0] / x, patch_size[1] / y), order=3)
    prediction = np.argmax(probability, axis=-1)

    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:

        image_data = np.moveaxis(image[:, :, :, 2].astype(np.float32), 0, -1)
        prediction_data = np.moveaxis(prediction.astype(np.float32), 0, -1)
        label_data = np.moveaxis(label.astype(np.float32), 0, -1)

        image_data = np.rot90(np.flip(image_data, axis=1), k=-1, axes=(0, 1))
        prediction_data = np.rot90(np.flip(prediction_data, axis=1), k=-1, axes=(0, 1))
        label_data = np.rot90(np.flip(label_data, axis=1), k=-1, axes=(0, 1))

        # Create Nifti images
        img_nifti = nib.Nifti1Image(image_data, np.eye(4))
        prd_nifti = nib.Nifti1Image(prediction_data, np.eye(4))
        lab_nifti = nib.Nifti1Image(label_data, np.eye(4))

        # Set spacing
        img_nifti.header["pixdim"][1:4] = [1, 1, 1]
        prd_nifti.header["pixdim"][1:4] = [1, 1, 1]
        lab_nifti.header["pixdim"][1:4] = [1, 1, 1]

        # Save the images
        img_nifti.to_filename(f"{test_save_path}/{case}_img.nii.gz")
        prd_nifti.to_filename(f"{test_save_path}/{case}_pred.nii.gz")
        lab_nifti.to_filename(f"{test_save_path}/{case}_gt.nii.gz")

    return metric_list


def inference(args, multimask_output, model, test_save_path=None):
    df = pd.read_csv(args.data_path + "/csv_files/" + args.stage + ".csv")

    # Create dictionary to store paths for each case
    case_slices = defaultdict(list)

    # Group paths by case ID
    for _, row in df.iterrows():
        case_id = row["id"]  # e.g., 'Case80'
        image_path = row["image_pth"]
        case_slices[case_id].append(image_path)

    # Sort paths within each case
    for case_id in case_slices:
        case_slices[case_id].sort(key=lambda x: int(x.split("slice")[-1].split(".")[0]))

    model.eval()
    metric_list = []

    for case_id, paths in case_slices.items():
        print(f"{case_id}: {len(paths)} slices")
        image_arr_list = []
        mask_arr_list = []
        un_arr_list = []
        for p in paths:
            with open(p, "rb") as file:
                image_arr = pickle.load(file)
            with open(p.replace(".pkl", "_segmentation.pkl"), "rb") as file:
                mask_arr = pickle.load(file)
            with open(p.replace(".pkl", "_uncertainty_10.pkl"), "rb") as file:
                un_arr = pickle.load(file)

            image_arr = np.clip(image_arr, HU_min, HU_max)
            image_arr = (image_arr - HU_min) / (HU_max - HU_min) * 255.0
            image_arr = np.float32(image_arr)
            image_arr = (image_arr - data_mean) / data_std
            image_arr = (image_arr - image_arr.min()) / (
                image_arr.max() - image_arr.min() + 0.00000001
            )

            mask_arr = np.float32(mask_arr)
            un_arr = average_pooling(un_arr.astype(np.float32))
            if args.num_classes == 12:
                mask_arr[mask_arr == 13] = 12
                class_to_name = {
                    1: "spleen",
                    2: "right kidney",
                    3: "left kidney",
                    4: "gallbladder",
                    5: "esophagus",
                    6: "liver",
                    7: "stomach",
                    8: "aorta",
                    9: "vena",
                    10: "vein",
                    11: "pancreas",
                    12: "adrenal gland",
                }

            image_arr_list.append(image_arr)
            mask_arr_list.append(mask_arr)
            un_arr_list.append(un_arr)

        image = np.expand_dims(np.stack(image_arr_list), axis=0)
        label = np.expand_dims(np.stack(mask_arr_list), axis=0)
        uncertainty = np.expand_dims(np.stack(un_arr_list), axis=0)
        case_name = case_id

        h, w = image.shape[2], image.shape[3]

        metric_i = test_single_volume(
            image,
            label,
            uncertainty,
            model,
            classes=args.num_classes,
            multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
        )

        metric_list.append(np.array(metric_i))
        logging.info(
            "idx %d case %s mean_dice %f" % (1, case_name, np.nanmean(metric_i, axis=0))
        )

    metric_list = np.nanmean(metric_list, axis=0)
    # for i in range(1, args.num_classes + 1):
    #     logging.info('Mean class %d name %s mean_dice %f' % (i, class_to_name[i], metric_list[i - 1]))

    performance = np.nanmean(metric_list, axis=0)
    std = np.nanstd(metric_list, axis=0)

    logging.info(
        f"Testing performance in best val model: mean_dice : {performance:.4f}, std: {std:.4f}" 
    )
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, "r") as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(": ")
        items_dict[key] = value
    return items_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapt_ckpt",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/project_results/MA_SAM/results-1/epoch_159.pth",
        help="The checkpoint after adaptation",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/synapseCT/Training/2D_all_5slice",
    )

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--img_size", type=int, default=256, help="Input image size of the network"
    )

    parser.add_argument("--stage", type=str, default="test", help="valid or test")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--is_savenii",
        action="store_true",
        help="Whether to save results during inference",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/PretrainedModel/sam_vit_h_4b8939.pth",
        help="Pretrained checkpoint",
    )
    parser.add_argument(
        "--vit_name", type=str, default="vit_b", help="Select one vit model"
    )
    parser.add_argument("--rank", type=int, default=32, help="Rank for FacT adaptation")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--module", type=str, default="sam_fact_tt_image_encoder")
    parser.add_argument(
        "--finetune_method",
        type=str,
        choices=["MA_SAM", "MA_UN_SAM"],
        default="MA_SAM",
        help="Finetune method for the model",
    )

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.output_dir = args.adapt_ckpt[:-4]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )

    pkg = import_module(args.module)
    if args.finetune_method == "MA_SAM":
        net = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()
    elif args.finetune_method == "MA_UN_SAM":
        net = pkg.Fact_tt_un_Sam(sam, args.rank, s=args.scale).cuda()

    assert args.adapt_ckpt is not None
    net.load_parameters(args.adapt_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, "test_log")
    os.makedirs(log_folder, exist_ok=True)

    if not os.path.exists("./testing_log"):
        os.mkdir("./testing_log")
    logging.basicConfig(
        filename="../testing_log/" + args.adapt_ckpt.split("/")[-2] + "_" + args.adapt_ckpt.split("/")[-1].split('.')[0] + "_log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = args.output_dir
    else:
        test_save_path = None
    inference(args, multimask_output, net, test_save_path)
