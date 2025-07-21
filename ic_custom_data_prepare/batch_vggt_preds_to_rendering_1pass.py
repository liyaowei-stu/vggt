import os, sys
import glob
import json
import argparse
import datetime
import random
import shutil

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
from vggt.utils.geometry import closed_form_inverse_se3


from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
    PerspectiveCameras
)


sys.path.append(os.getcwd())
# from ic_custom_data_prepare.navi_datasets import loader as navi_loader
from ic_custom_data_prepare.navi_datasets_fixed_seed import loader as navi_loader

from ic_custom_data_prepare.utils import calculate_extrinsic_correction, excute_extrinsic_correction, calculate_intrinsic_correction, excute_intrinsic_correction

def np_depth_to_colormap(depth, min_conf=-0.9):
    """ 
    Args:
    depth: [B, N, H, W, 1] or [B, N, H, W]  or [B, H, W]
    
    Returns:
    depth_color: [B, N, H, W, 3] or [B, H, W, 3]
    depth_normalized: [B, N, H, W, 1] or [B, H, W, 1]
    """

    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    if depth.ndim == 4:
        dpt_ndim = 4
        b, n, h, w = depth.shape
        depth = depth.reshape(-1, h, w)
    elif depth.ndim == 3:
        dpt_ndim = 3
        b, h, w = depth.shape
        depth = depth.reshape(-1, h, w)
    else:
        raise ValueError(f"Depth dimension is not supported: {depth.ndim}")

    depth_colors = []
    depth_normalized = []

    for dpt in depth:
        dpt_normalized = np.zeros(dpt.shape)
        valid_mask_dp = dpt > min_conf # valid

        if valid_mask_dp.sum() > 0:
            d_valid = dpt[valid_mask_dp]
            min_val = d_valid.min()
            max_val = d_valid.max()
            if max_val > min_val:  # Avoid division by zero
                dpt_normalized[valid_mask_dp] = (d_valid - min_val) / (max_val - min_val)
            else:
                # If all values are the same, set normalized value to 0.5
                dpt_normalized[valid_mask_dp] = 0.5

            dpt_np = (dpt_normalized * 255).astype(np.uint8)
            dpt_color = cv2.applyColorMap(dpt_np, cv2.COLORMAP_JET)
            depth_colors.append(dpt_color)
            depth_normalized.append(dpt_normalized)
        else:
            print('!!!! No depth projected !!!')
            dpt_color = np.zeros((dpt.shape[0], dpt.shape[1], 3), dtype=np.uint8)
            dpt_normalized = np.zeros(dpt.shape, dtype=np.float32)
            depth_colors.append(dpt_color)
            depth_normalized.append(dpt_normalized)

    depth_colors = np.stack(depth_colors, axis=0)
    depth_normalized = np.stack(depth_normalized, axis=0)

    if dpt_ndim == 4:
        depth_colors = depth_colors.reshape(b, n, h, w, 3)
        depth_normalized = depth_normalized.reshape(b, n, h, w, 1)
    elif dpt_ndim == 3:
        depth_colors = depth_colors.reshape(b, h, w, 3)
        depth_normalized = depth_normalized.reshape(b, h, w, 1)

    return depth_colors, depth_normalized



def define_rasterizer_renderer(cameras, image_size=(392, 518), radius=0.003, points_per_pixel=10, bin_size=None):
    """
    Define the rasterizer and renderer for the point cloud
    """
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel,
        bin_size=bin_size,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(255, 255, 255))
    )
    
    return rasterizer, renderer


def run_vggt_on_images(images, model):
    """
    images: [B, S, 3, H, W], tensor
    """
    with torch.cuda.amp.autocast(dtype=dtype):

        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

    return predictions


def get_points_from_mask(points_flat, colors_flat, conf_flat, masks):
    """
    Args:
    points_flat: (N, 3), numpy array
    colors_flat: (N, 3), numpy array
    conf_flat: (N,), numpy array
    masks: (B, S, H, W), numpy array
    """
    masks_flat = masks.reshape(-1)
    masks_flat_bool = masks_flat.astype(bool)
    points_flat = points_flat[masks_flat_bool]  # shape: (n, 3)
    conf_flat = conf_flat[masks_flat_bool]  # shape: (n,)
    colors_flat = colors_flat[masks_flat_bool]   # shape: (n, 3)
    
    return points_flat, colors_flat, conf_flat


def get_points_from_predictions(predictions, use_point_map: bool = False, max_points: int = 1000000, conf_threshold: float = 20.0, conf_threshold_value: float = 2.0, apply_mask: bool = False, recon_intrinsic: torch.Tensor = None, recon_extrinsic: torch.Tensor = None):
    """
    Args:
    predictions: dict
    use_point_map: bool, if True, use point map, otherwise use depth map
    max_points: int, max points to sample
    conf_threshold: float, confidence threshold
    apply_mask: bool, if True, apply mask to the points
    device: str, device to use

    Returns:
        points_flat: (N, 3), numpy array
        colors_flat: (N, 3), numpy array
        conf_flat: (N,), numpy array
    """
        
    images = predictions["images"]  # (B, S, 3, H, W)
    world_points_map = predictions["world_points"]  # (B, S, H, W, 3)
    conf_map = predictions["world_points_conf"]  # (B, S, H, W)
    depth_map = predictions["depth"]  # (B, S, H, W, 1) 
    depth_conf = predictions["depth_conf"]  # (B, S, H, W)
    extrinsics_cam = predictions["extrinsic"]  # (B, S, 3, 4)
    intrinsics_cam = predictions["intrinsic"]  # (B, S, 3, 3)
    masks = predictions["masks"]  # (B, S, H, W)

    if recon_intrinsic is not None and recon_extrinsic is not None:
        extrinsics_cam = recon_extrinsic.cpu().numpy()
        intrinsics_cam = recon_intrinsic.cpu().numpy()

    bsz, s, h, w, _ = world_points_map.shape

    points_flat_list = []
    colors_flat_list = []
    conf_flat_list = []

    conf_mean_list = []

    
    for i in range(bsz):
        if not use_point_map:
            world_points = unproject_depth_map_to_point_map(depth_map[i], extrinsics_cam[i], intrinsics_cam[i])
            conf = depth_conf[i]
        else:
            world_points = world_points_map[i]
            conf = conf_map[i]


        points_flat = world_points.reshape(-1, 3)
        colors_flat = (images[i].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = conf.reshape(-1)

        mask = masks[i]
        
        if apply_mask:
            points_flat, colors_flat, conf_flat = get_points_from_mask(points_flat, colors_flat, conf_flat, mask)

        if points_flat.shape[0] > max_points:
            # get the top max_points points
            sorted_indices = np.argsort(conf_flat)[::-1]
            sorted_indices = sorted_indices[:max_points]
            points_flat = points_flat[sorted_indices]
            colors_flat = colors_flat[sorted_indices]
            conf_flat = conf_flat[sorted_indices]

        # Filter points based on confidence threshold
        conf_mean_list.append(np.mean(conf_flat))
        threshold_val = np.percentile(conf_flat, conf_threshold)
        threshold_val = max(threshold_val, conf_threshold_value)
        conf_mask = conf_flat > threshold_val
        points_flat = points_flat[conf_mask]
        colors_flat = colors_flat[conf_mask]
        conf_flat = conf_flat[conf_mask]

        points_flat_list.append(points_flat)
        colors_flat_list.append(colors_flat)
        conf_flat_list.append(conf_flat)

    return points_flat_list, colors_flat_list, conf_flat_list, conf_mean_list


def get_depth_from_mask(depth_map, mask):
    """
    Args:
    depth_map: (B, H, W, 1), numpy array
    mask: (B, H, W), numpy array

    Returns:
        depth_fill: (B, H, W, 1), numpy array
    """

    depth_fill = np.zeros(depth_map.shape)
    depth_fill_flat = depth_fill.reshape(-1, 1)

    mask_flat_bool = mask.reshape(-1).astype(bool)

    depth_flat = depth_map.reshape(-1, 1)[mask_flat_bool]
    
    if len(depth_flat) == 0 or (depth_flat.max() - depth_flat.min()) < 1e-4:
        return depth_map

    depth_flat_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min())

    depth_fill_flat[mask_flat_bool] = depth_flat_norm

    depth_fill = depth_fill_flat.reshape(depth_map.shape)

    return depth_fill



def get_depth_fill_from_predictions(predictions,  max_points: int = 1000000, conf_threshold: float = 20.0, conf_threshold_value: float = 2.0, apply_mask: bool = False):
    """"
    This function is used to get the depth fill from the predictions
    Due to the different methods of point selection, it is handled separately from get_points_from_predictions.

    apply_mask will be forbidden in this function

    Args:
    predictions: dict
    max_points: int, max points to sample
    conf_threshold: float, confidence threshold
    apply_mask: bool, if True, apply mask to the points

    Returns:
        depth_fill: (B, S, H, W, 1), numpy array
    """
    depths_map = predictions["depth"]  # (B, S, H, W, 1) 
    depths_conf = predictions["depth_conf"]  # (B, S, H, W)
    masks = predictions["masks"]  # (B, S, H, W)

    bsz, s, h, w, _ = depths_map.shape

    depth_fill_list = []
    for i in range(bsz):
        mask = masks[i]
        depth_map = depths_map[i]
        depth_conf = depths_conf[i]


        if apply_mask:
            depth_map = get_depth_from_mask(depth_map, mask)
            depth_conf = get_depth_from_mask(depth_conf, mask)
        
        depth_conf_flat = depth_conf.reshape(-1)
        depth_fill = np.zeros(depth_map.shape)
        depth_fill_flat = depth_fill.reshape(-1, 1)
        depth_flat = depth_map.reshape(-1, 1)

        if depth_flat.shape[0] > max_points:
            # get the top max_points points
            sorted_indices = np.argsort(depth_conf_flat)[::-1]
            sorted_indices = sorted_indices[:max_points]
            depth_flat = depth_flat[sorted_indices]
            depth_conf_flat = depth_conf_flat[sorted_indices]

        # Filter points based on confidence threshold
        threshold_val = np.percentile(depth_conf_flat, conf_threshold)
        threshold_val = max(threshold_val, conf_threshold_value)
        conf_mask = depth_conf_flat > threshold_val
        depth_flat = depth_flat[conf_mask]
        
        # Handle the case where depth_flat is empty (zero-size array)
        if depth_flat.size > 0:
            depth_flat_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min())
            conf_mask_sorted_indices = sorted_indices[conf_mask]
            depth_fill_flat[conf_mask_sorted_indices] = depth_flat_norm
            depth_fill = depth_fill_flat.reshape(depth_map.shape)
        else:
            depth_fill = depth_map

        depth_fill_list.append(depth_fill)

    depth_fill =  np.stack(depth_fill_list, axis=0)

    return depth_fill


        
def convert_opencv_to_pytorch3d_c2w(c2w):
    """
    c2w: (N, 3, 4), column major
    return: (N, 3, 4)
    """
    # rotate axis
    opencv_R, T = c2w[:, :3, :3], c2w[:, :3, 3]
    pytorch_three_d_R = np.stack([-opencv_R[:, :, 0], -opencv_R[:, :, 1], opencv_R[:, :, 2]], 2)
    # pytorch_three_d_R = np.stack([-opencv_R[:, 0, :], -opencv_R[:, 1, :], opencv_R[:, 2, :]], 1)
    
    # convert to w2c
    new_c2w = np.concatenate([pytorch_three_d_R, T[:, :, None]], axis=2) # 3*4
    return new_c2w



@torch.no_grad()
def main(
    args, 
    ):

    import ipdb; ipdb.set_trace()

    data_loader = navi_loader(train_batch_size=args.batch_size, num_workers=0, meta_path=args.meta_path, data_dir=args.data_dir)

    for i, data in enumerate(tqdm(data_loader)):

        sample_render_intrinsic, sample_render_extrinsic, sample_render_image, sample_render_mask, sample_render_idxs, sample_recon_intrinsic, sample_recon_extrinsic, sample_recon_images, sample_recon_masks, sample_recon_idxs, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, multi_view_paths = data
        bsz, s, _, h, w = sample_recon_images.shape


        ## recon and filter points
        pred_dict = {}
        pred_dict["images"] = sample_recon_images.cpu().numpy()
        pred_dict["extrinsic"] = sample_recon_extrinsic.cpu().numpy()
        pred_dict["intrinsic"] = sample_recon_intrinsic.cpu().numpy()
        pred_dict["masks"] = sample_recon_masks.cpu().numpy()
        pred_dict["depth"] = sample_recon_depths.cpu().numpy()
        pred_dict["depth_conf"] = sample_recon_depth_confs.cpu().numpy()
        pred_dict["world_points"] = sample_recon_world_points.cpu().numpy()
        pred_dict["world_points_conf"] = sample_recon_world_points_confs.cpu().numpy()

        
        points_flat, colors_flat, conf_flat, conf_mean = get_points_from_predictions(pred_dict, args.use_point_map, args.max_points, args.conf_threshold, args.conf_threshold_value, args.apply_mask)
        depth_fill = get_depth_fill_from_predictions(pred_dict, args.max_points, args.conf_threshold, args.conf_threshold_value, args.apply_mask)

        depth_color, depth_normalized = np_depth_to_colormap(depth_fill)
     
        # squeeze the num_views dimension
        if len(sample_render_extrinsic.shape) == 4:
            sample_render_extrinsic = sample_render_extrinsic.squeeze(1)
        if len(sample_render_intrinsic.shape) == 4:
            sample_render_intrinsic = sample_render_intrinsic.squeeze(1)


        ## coordinate system conversion
        sample_render_c2w = closed_form_inverse_se3(sample_render_extrinsic)
        sample_render_c2w = sample_render_c2w[:, :3, :]

        
        # Save point cloud for fast visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # trimesh.PointCloud(points_flat[0], colors=colors_flat[0]).export(f"{timestamp}_points_max_points_{args.max_points}_conf_{args.conf_threshold}_use_point_map_{args.use_point_map}_apply_mask_{args.apply_mask}.ply")

        # move the scene to the center
        scene_centers = []
        points_flat_centered = []
        for points_flat in points_flat:
            scene_center = np.mean(points_flat, axis=0)
            points_flat_centered.append(points_flat - scene_center)
            scene_centers.append(scene_center)

        scene_centers = np.stack(scene_centers, axis=0)

        sample_render_c2w[..., -1] -= scene_centers

        sample_render_c2w = convert_opencv_to_pytorch3d_c2w(sample_render_c2w)
        sample_render_w2c = closed_form_inverse_se3(sample_render_c2w)
        sample_render_w2c_R, sample_render_w2c_T = sample_render_w2c[:, :3, :3], sample_render_w2c[:, :3, 3]

        # pytorch3d is row major, so we need to transpose the R
        sample_render_w2c_R = torch.from_numpy(sample_render_w2c_R.transpose(0, 2, 1)).float()
        sample_render_w2c_T = torch.from_numpy(sample_render_w2c_T).float()


        fx, fy = sample_render_intrinsic[:, 0, 0], sample_render_intrinsic[:, 1, 1]
        ux, uy = sample_render_intrinsic[:, 0, 2], sample_render_intrinsic[:, 1, 2]

       
        image_size = [ [sample_recon_images.shape[-2], sample_recon_images.shape[-1]] for _ in range(bsz)]

        
        fcl_screen = torch.stack((fx, fy), dim=-1)
        prp_screen = torch.stack((ux, uy), dim=-1)


        # ## squeeze the num_views dimension
        cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False)

        rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=args.radius, points_per_pixel=args.points_per_pixel, bin_size=args.bin_size)

        points_flat_centered = [torch.from_numpy(points_flat_centered[i]).float().to(device) for i in range(bsz)]
        colors_flat = [torch.from_numpy(colors_flat[i]).float().to(device) for i in range(bsz)]
        point_cloud = Pointclouds(points=points_flat_centered, features=colors_flat)

        new_images, fragments = renderer(point_cloud)

        new_depths = fragments.zbuf[:, :, :, 0].cpu().numpy()
        new_depth_color, new_depth_normalized = np_depth_to_colormap(new_depths)

        ## save info
        for i, new_image in enumerate(new_images):
            
            save_dir = f"{args.output_dir}/conf_{args.conf_threshold}_{len(sample_recon_images[0])}_recon_num/{multi_view_paths[i]}"

            if os.path.exists(f"{save_dir}/renders"):
                shutil.rmtree(f"{save_dir}/renders")
            os.makedirs(f"{save_dir}/renders", exist_ok=True)

            ## save renders
            new_image = new_image.cpu().numpy()
            new_image = new_image.astype(np.uint8)
            new_image = Image.fromarray(new_image)
            save_path = f"{save_dir}/renders/rasterized/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            new_image.save(save_path)

            save_path = f"{save_dir}/renders/gt/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(sample_render_image[i], save_path)

            save_path = f"{save_dir}/renders/rasterized_depth_color/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, new_depth_color[i])
            save_path = f"{save_dir}/renders/rasterized_depth/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, (new_depth_normalized[i] * 255).astype(np.uint8))

            ## save recon
            sample_recon_idx = sample_recon_idxs[i]
            sample_recon_image = sample_recon_images[i]
            dpt_color = depth_color[i]
            dpt_normalized = depth_normalized[i]

            ## remove the recon folder if it exists
            if os.path.exists(f"{save_dir}/recon"):
                shutil.rmtree(f"{save_dir}/recon")

            for j in range(len(sample_recon_image)):
                save_path = f"{save_dir}/recon/images/sample_idx_{sample_recon_idx[j].item()}.png"
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image(sample_recon_image[j], save_path)

                save_path = f"{save_dir}/recon/depth_color/sample_idx_{sample_recon_idx[j].item()}.png"
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, dpt_color[j])

                save_path = f"{save_dir}/recon/depth/sample_idx_{sample_recon_idx[j].item()}.png"
    
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, (dpt_normalized[j] * 255).astype(np.uint8))

            save_path = f"{save_dir}/recon/points/points.ply"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trimesh.PointCloud(points_flat_centered[i].cpu().numpy(), colors=(colors_flat[i].cpu().numpy()).astype(np.uint8)).export(save_path)

            with open(f"{save_dir}/recon/conf_mean.txt", "w") as f:
                f.write(f"conf_mean: {conf_mean[i]}")

            # print("="*100)

    print("Process done!")



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_point_map", type=bool, default=False)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--conf_threshold", type=float, default=20.0)
    parser.add_argument("--conf_threshold_value", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--points_per_pixel", type=int, default=20)
    parser.add_argument("--bin_size", type=int, default=None)
    parser.add_argument("--apply_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()



if __name__ == "__main__":
    args = parser_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    args.device = device
    args.dtype = dtype

    main(args)
