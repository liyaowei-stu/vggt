import os, sys
import glob
import json
import argparse
import datetime


import numpy as np
from einops import rearrange
from PIL import Image

import torch
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


def save_points_dict(output_path, points, colors, points_conf):

    
    # Convert tensors to numpy arrays before saving
    if isinstance(points, torch.Tensor):
        points_dict_np = {
            "verts": points.float().detach().cpu().numpy(),
            "rgb": colors.float().detach().cpu().numpy().astype(np.uint8),
            "conf": points_conf.float().detach().cpu().numpy()
        }
    else:
        points_dict_np = {
            "verts": points,
            "rgb": colors,
            "conf": points_conf
        }
    
    np.savez(output_path, **points_dict_np)


@torch.no_grad()
def main(
    image_path, 
    model, 
    use_point_map: bool = False, 
    max_points: int = 1000000, 
    conf_threshold: float = 20.0
    ):

    # import ipdb; ipdb.set_trace()

    images = load_and_preprocess_images(image_path).to(device)

    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension


        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        np.savez(f"output/test_data_soldier_predictions_{timestamp}.npz", **predictions)


        b, s, c, h, w = images.shape
        aggregated_tokens_list, ps_idx = model.aggregator(images)


        if not use_point_map:

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            point_map = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                        extrinsic.squeeze(0), 
                                                                        intrinsic.squeeze(0))
            point_map = torch.from_numpy(point_map).to(depth_conf)
            point_map = rearrange(point_map, "(b s) c h w -> b s c h w", b=b, s=s)
            point_conf = depth_conf
        else:
            # Predict Point Maps
            point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            point_map = rearrange(point_map, "(b s) c h w -> b s c h w", b=b, s=s)
            

    # Flatten
    colors =  images.permute(0, 1, 3, 4, 2)  # now (B, S, H, W, 3)
    # colors_pil = Image.fromarray((colors.cpu().numpy() * 255).astype(np.uint8))
    colors_flat = (colors.reshape(b, -1, 3))

    points_flat = point_map.reshape(b, -1, 3)
    point_conf_flat = point_conf.reshape(b, -1)

    pts_b, pts_n, pts_c = points_flat.shape
    
    if pts_n > max_points:
        # Sort points by confidence
        sorted_indices = torch.argsort(point_conf_flat, dim=1, descending=True)
        # Select top max_points points
        sorted_indices = sorted_indices[:, :max_points]
        
        # Create batch indices for proper indexing
        batch_indices = torch.arange(b, device=device)
            
        # Index using batch indices and sorted indices
        points_flat = points_flat[batch_indices, sorted_indices]  # Shape: (b, max_points, 3)
        colors_flat = colors_flat[batch_indices, sorted_indices]  # Shape: (b, max_points, 3)
        point_conf_flat = point_conf_flat[batch_indices, sorted_indices]  # Shape: (b, max_points)


    # Filter points based on confidence threshold
    # Convert tensor to CPU numpy array before computing percentile
    threshold_val = torch.tensor(np.percentile(point_conf_flat.cpu().numpy(), conf_threshold)).to(point_conf_flat.device)
    conf_mask = point_conf_flat > threshold_val
    points_flat = points_flat[conf_mask]
    colors_flat = colors_flat[conf_mask]
    point_conf_flat = point_conf_flat[conf_mask]

    # Compute scene center and recenter
    scene_center = points_flat.mean(axis=0)
    points_centered = points_flat - scene_center


    #  # Save the point cloud data as npz file
    # viz_frame = False
    # if viz_frame:
    #     frame_idx = -2
    #     colors_frame = colors[:, frame_idx]
    #     colors_frame_flat = colors_frame.reshape(-1, 3)
    #     points_frame = point_map[:, frame_idx]
    #     point_conf_frame = point_conf[:, frame_idx]
    #     points_frame_flat = points_frame.reshape(-1, 3)
    #     point_conf_frame_flat = point_conf_frame.reshape(-1)
    #     viz_points_flat, viz_colors_flat, viz_point_conf_flat = points_frame_flat, colors_frame_flat, point_conf_frame_flat
    #     viz_points_flat_centered = points_frame_flat - points_frame_flat.mean(axis=0)
    # else:
    #     viz_points_flat_centered, viz_colors_flat, viz_point_conf_flat = points_centered, colors_flat, point_conf_flat
    # output_dir = "output"
    # os.makedirs(output_dir, exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_path = os.path.join(output_dir, f"point_cloud_{timestamp}_conf_{conf_threshold}.npz")
    # save_points_dict(output_path, viz_points_flat_centered, viz_colors_flat * 255, viz_point_conf_flat)

    
    # define camera and rasterization to render scene
    point_cloud = Pointclouds(points=[points_centered], features=[colors_flat])
    # Initialize a camera, R,T is world to camera, R:1*3*3, T:1*3*1
    R, T = look_at_view_transform(1, 270, 180)

    # extrinsic[..., -1] += torch.bmm(extrinsic[0,:,:,:3], scene_center.unsqueeze(-1).repeat(25,1,1)).squeeze(-1).unsqueeze(0)
    # R, T = extrinsic[0,0,:,:3], extrinsic[0,0,:,3]
    # R = R.unsqueeze(0)
    # T = T.unsqueeze(0)

    # K = torch.zeros((b, 4, 4), dtype=torch.float32, device=R.device)
    # ones = torch.ones((b), dtype=torch.float32, device=R.device)
    # K[:, 3, 3] = ones

    # K[:, :3, :3] = intrinsic[0,0,:,:3]

    # cameras = FoVOrthographicCameras(device=device, R=R, T=T, zfar=0.1)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=(392, 518), 
        radius = 0.003,
        points_per_pixel = 10,
        bin_size=0,
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )


    rendered_images = renderer(point_cloud)

    rendered_images_pil_list = [Image.fromarray((rendered_image.cpu().numpy() * 255).astype(np.uint8)) for rendered_image in rendered_images]
    
    print("Finished rendering")
    # import ipdb; ipdb.set_trace()


    return rendered_images_pil_list



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--use_point_map", type=bool, default=False)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--conf_threshold", type=float, default=20.0)

  
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parser_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    image_path = sorted(glob.glob(os.path.join(args.image_path, "*")))

    model = VGGT.from_pretrained(args.model).to(device)

    main(image_path, model, args.use_point_map, args.max_points, args.conf_threshold)
