import os, sys
import glob
import json
import argparse
import datetime
import random

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
from ic_custom_data_prepare.navi_datasets import loader as navi_loader

from ic_custom_data_prepare.utils import calculate_extrinsic_correction, excute_extrinsic_correction, calculate_intrinsic_correction, excute_intrinsic_correction

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


def get_points_from_predictions(predictions, use_point_map: bool = False, max_points: int = 1000000, conf_threshold: float = 20.0, apply_mask: bool = False, recon_intrinsic: torch.Tensor = None, recon_extrinsic: torch.Tensor = None):
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

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trimesh.PointCloud(points_flat, colors=colors_flat).export(f"{timestamp}_before_filter.ply")

        sorted_indices = np.argsort(conf_flat)[::-1]
        if points_flat.shape[0] > max_points:
            # get the top max_points points
            sorted_indices = sorted_indices[:max_points]
            points_flat = points_flat[sorted_indices]
            colors_flat = colors_flat[sorted_indices]
            conf_flat = conf_flat[sorted_indices]
            depth_flat = depth_flat[sorted_indices]

        # Filter points based on confidence threshold
        threshold_val = np.percentile(conf_flat, conf_threshold)
        threshold_val = min(threshold_val, 2.0)
        conf_mask = conf_flat > threshold_val
        points_flat = points_flat[conf_mask]
        colors_flat = colors_flat[conf_mask]
        conf_flat = conf_flat[conf_mask]

        trimesh.PointCloud(points_flat, colors=colors_flat).export(f"{timestamp}_after_filter.ply")

        points_flat_list.append(points_flat)
        colors_flat_list.append(colors_flat)
        conf_flat_list.append(conf_flat)

    return points_flat_list, colors_flat_list, conf_flat_list


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
    model,
    ):

    import ipdb; ipdb.set_trace()

    data_loader = navi_loader(train_batch_size=args.batch_size, num_workers=0, meta_path=args.meta_path, data_dir=args.data_dir)

    for i, data in enumerate(data_loader):
        # sample_render_extrinsic: (B, 3, 4)
        # sample_render_intrinsic: (B, 3, 3)
        # sample_render_idxs: (B,)
        # sample_recon_images: (B, S, 3, H, W)
        # sample_recon_masks: (B, S, H, W)
        # sample_recon_idxs: (B, S)
        sample_render_intrinsic, sample_render_extrinsic, sample_render_image, sample_render_mask, sample_render_idxs, sample_recon_intrinsic, sample_recon_extrinsic, sample_recon_images, sample_recon_masks, sample_recon_idxs, multi_view_paths = data
        bsz, s, _, h, w = sample_recon_images.shape

        sample_render_extrinsic_uncorrected = sample_render_extrinsic.clone()
        sample_render_intrinsic_uncorrected = sample_render_intrinsic.clone()

        sample_render_extrinsic = sample_render_extrinsic.unsqueeze(1) # B,3,4 - > B,1,3,4
        sample_render_intrinsic = sample_render_intrinsic.unsqueeze(1) # B,3,3 - > B,1,3,3

        ## above is dataloader, this is the main function
        sample_recon_images = sample_recon_images.to(args.device)
        pred_dict = run_vggt_on_images(sample_recon_images, model)
        pred_dict["masks"] = sample_recon_masks.cpu().numpy()

        
        points_flat, colors_flat, conf_flat = get_points_from_predictions(pred_dict, args.use_point_map, args.max_points, args.conf_threshold, args.apply_mask)

        # ## align 2 corrdanate system
        recon_extrinsics_2 = torch.from_numpy(pred_dict["extrinsic"]).to(sample_recon_extrinsic.device)  # (B, S, 3, 4)
        recon_intrinsics_2 = torch.from_numpy(pred_dict["intrinsic"]).to(sample_recon_extrinsic.device)  # (B, S, 3, 3)

        # # Calculate the correction matrix between recon_extrinsics_2 and sample_recon_extrinsic
        # # The correction matrix transforms sample_recon_extrinsic to match recon_extrinsics_2
        correct_extrinsic_matrix = calculate_extrinsic_correction(sample_recon_extrinsic, recon_extrinsics_2)
        sample_render_extrinsic = excute_extrinsic_correction(sample_render_extrinsic, correct_extrinsic_matrix)

        # # valid: Calculate the error between the corrected extrinsic and recon_extrinsics_2
        corrected_extrinsic = excute_extrinsic_correction(sample_recon_extrinsic, correct_extrinsic_matrix)        
        error = torch.norm(corrected_extrinsic - recon_extrinsics_2)
        ori_error = torch.norm(sample_recon_extrinsic - recon_extrinsics_2)
        print(f"Correction matrix error: {error.item()}, ori error: {ori_error.item()}")

        
        # # Calculate the intrinsic correction scale between recon_intrinsics_2 and sample_recon_intrinsic
        scale_x, scale_y, _ = calculate_intrinsic_correction(sample_recon_intrinsic, recon_intrinsics_2)
        sample_render_intrinsic = excute_intrinsic_correction(sample_render_intrinsic, scale_x, scale_y)
        
        # # valid: apply this scale to adjust points or camera parameters if needed
        corrected_intrinsic = excute_intrinsic_correction(sample_recon_intrinsic, scale_x, scale_y)
        error = torch.norm(corrected_intrinsic - recon_intrinsics_2)
        ori_error = torch.norm(sample_render_intrinsic - recon_intrinsics_2)
        print(f"Intrinsic correction error: {error.item()}, ori error: {ori_error.item()}")

     
        # squeeze the num_views dimension
        if len(sample_render_extrinsic.shape) == 4:
            sample_render_extrinsic = sample_render_extrinsic.squeeze(1)
        if len(sample_render_intrinsic.shape) == 4:
            sample_render_intrinsic = sample_render_intrinsic.squeeze(1)

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
        cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False,)
        rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=args.radius, points_per_pixel=args.points_per_pixel, bin_size=args.bin_size)

        points_flat_centered = [torch.from_numpy(points_flat_centered[i]).float().to(device) for i in range(bsz)]
        colors_flat = [torch.from_numpy(colors_flat[i]).float().to(device) for i in range(bsz)]
        point_cloud = Pointclouds(points=points_flat_centered, features=colors_flat)

        new_images, fragments = renderer(point_cloud)

        for i, new_image in enumerate(new_images):
            new_image = new_image.cpu().numpy()
            new_image = new_image.astype(np.uint8)
            new_image = Image.fromarray(new_image)
            new_image.save(f"{timestamp}_new_image_{i}.png")

            ## save sample recon images and gt images
            save_image(sample_recon_images[i], f"{timestamp}_sample_recon_images_{i}.png")
            save_image(sample_render_image[i], f"{timestamp}_gt_image_sample_{i}.png")


        ## vggt preds parameters valid
        sample_render_extrinsic =  pred_dict['extrinsic'][:, 1]
        sample_render_intrinsic = torch.from_numpy(pred_dict['intrinsic'][:, 1]).float()
        sample_render_c2w = closed_form_inverse_se3(sample_render_extrinsic)
        sample_render_c2w = sample_render_c2w[:, :3, :]

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
        cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False,)
        rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=args.radius, points_per_pixel=args.points_per_pixel, bin_size=args.bin_size)

        new_images, fragments = renderer(point_cloud)

        for i, new_image in enumerate(new_images):
            new_image = new_image.cpu().numpy()
            new_image = new_image.astype(np.uint8)
            new_image = Image.fromarray(new_image)
            new_image.save(f"{timestamp}_new_image_vggt_gt_render{i}.png")


        ## uncorrected parameters valid
        sample_render_intrinsic = sample_render_intrinsic_uncorrected
        sample_render_extrinsic = sample_render_extrinsic_uncorrected
        sample_render_c2w = closed_form_inverse_se3(sample_render_extrinsic)
        sample_render_c2w = sample_render_c2w[:, :3, :]

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
        cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False,)
        rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=args.radius, points_per_pixel=args.points_per_pixel, bin_size=args.bin_size)

        new_images, fragments = renderer(point_cloud)

        for i, new_image in enumerate(new_images):
            new_image = new_image.cpu().numpy()
            new_image = new_image.astype(np.uint8)
            new_image = Image.fromarray(new_image)
            new_image.save(f"{timestamp}_new_image_render_uncorrected{i}.png")





    print("Process done!")



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--use_point_map", type=bool, default=False)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--conf_threshold", type=float, default=20.0)
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

    model = VGGT.from_pretrained(args.model).to(device)

    main(args, model)
