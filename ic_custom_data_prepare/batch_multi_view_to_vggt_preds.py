import os, sys
import glob
import json
import argparse
import datetime


import numpy as np
from einops import rearrange
from PIL import Image
import trimesh
import cv2
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)


def np_depth_to_colormap(depth, min_conf=-0.9):
    """ depth: [H, W] """
    depth_normalized = np.zeros(depth.shape)

    valid_mask_dp = depth > min_conf # valid

    if valid_mask_dp.sum() > 0:
        d_valid = depth[valid_mask_dp]
        depth_normalized[valid_mask_dp] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())

        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_color, depth_normalized



def run_vggt(images_path, masks_path, model):
    ## only save predictions, not to process the predictions
    images = load_and_preprocess_images(images_path, mode="pad").to(device)

    masks = load_and_preprocess_images(masks_path, mode="pad", is_mask=True)[:,0,:,:]

    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension

        predictions = model(images)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        # predictions["extrinsic"] = extrinsic
        # predictions["intrinsic"] = intrinsic
        predictions["masks"] = masks
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                if predictions[key].shape[0] == 1:
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
                else:
                    predictions[key] = predictions[key].cpu().numpy()  # remove batch dimension and convert to numpy


        # ### 1. process world points
        # world_points_map = predictions["world_points"]  # (S, H, W, 3)
        # world_points_conf = predictions["world_points_conf"]  # (S, H, W)
        # colors = predictions["images"].transpose(0, 2, 3, 1)

        # # Extract points and confidence values based on masks
        # S, H, W, _ = world_points_map.shape
        
        # # Flatten the world points and confidence maps
        # world_points_flat = world_points_map.reshape(-1, 3)
        # world_points_conf_flat = world_points_conf.reshape(-1)
        # colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        
        # # Flatten the masks and use them to filter points
        # masks_flat = masks.reshape(-1)
        # masks_flat_bool = masks_flat.astype(bool)
        
        # # Apply mask to get only the points and confidence values within the mask
        # masked_world_points_flat = world_points_flat[masks_flat_bool]  # shape: (n, 3)
        # masked_conf_flat = world_points_conf_flat[masks_flat_bool]  # shape: (n,)
        # masked_colors_flat = colors_flat[masks_flat_bool]  # shape: (n, 3)

        # # trimesh.PointCloud(masked_world_points_flat, colors=masked_colors_flat).export(f"test.ply")
        
        # # Add the masked points and confidence to predictions
        # predictions["world_points"] = masked_world_points_flat
        # predictions["world_points_conf"] = masked_conf_flat
        # # predictions["world_points_colors"] = masked_colors_flat

        # ### 2. process depth map
        # depth_map = predictions["depth"]  # (S, H, W, 1)
        # depth_conf = predictions["depth_conf"]  # (S, H, W)

        # masks_bool = masks.astype(bool)
        # depth_map_fill = np.zeros(depth_map.shape)

        # valid_depth_map = depth_map[masks_bool]
        # valid_depth_map_norm = (valid_depth_map - valid_depth_map.min()) / (valid_depth_map.max() - valid_depth_map.min())

        # depth_map_fill[masks_bool] = valid_depth_map_norm
        # depth_map_valid = depth_map_fill.reshape(depth_map.shape)

        # depth_color, depth_normalized = np_depth_to_colormap(depth_map_valid[0, :, :, 0])
        # cv2.imwrite(f"depth_color.png", depth_color.astype(np.uint8))
        # cv2.imwrite(f"depth_normalized.png", (depth_normalized*255).astype(np.uint8))

    return predictions


@torch.no_grad()
def main(
    args, 
    model, 
    ):


    meta_info = json.load(open(args.meta_path))
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for item in tqdm(meta_info):
        object_id = item["id"]
        multiview_path = item["multiview_img_path"]
        video_path = item["video_path"]

        for image_dir in tqdm(multiview_path):
            images_path = sorted(glob.glob(os.path.join(data_dir, image_dir, "images", "**")))
            masks_path = sorted(glob.glob(os.path.join(data_dir, image_dir, "masks", "**")))
            predictions = run_vggt(images_path, masks_path, model)

            scene_camera_name = image_dir.split("/")[-1]
            output_path = os.path.join(output_dir, object_id, scene_camera_name, "vggt_predictions.npz")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, **predictions)

        for video_dir in tqdm(video_path):
            images_path = sorted(glob.glob(os.path.join(data_dir, video_dir, "images", "**")))
            masks_path = sorted(glob.glob(os.path.join(data_dir, video_dir, "masks", "**")))
            scene_camera_name = video_dir.split("/")[-1]
            if len(images_path) > 200:
                # Uniformly sample 200 frames from 0 to len(images_path)
                indices = np.linspace(0, len(images_path) - 1, 200, dtype=int)

                output_path = os.path.join(output_dir, object_id, scene_camera_name, "sampled_indices.txt")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    for i in indices:
                        f.write(f"{os.path.basename(images_path[i]).split('.')[0]}\n") 

                images_path = [images_path[i] for i in indices]
                masks_path = [masks_path[i] for i in indices]

            predictions = run_vggt(images_path, masks_path, model)

            output_path = os.path.join(output_dir, object_id, scene_camera_name, "vggt_predictions.npz")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, **predictions)

    print("Process done!")



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B")
    return parser.parse_args()



if __name__ == "__main__":
    args = parser_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    model = VGGT.from_pretrained(args.model).to(device)

    main(args, model)
