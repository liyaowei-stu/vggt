import os, sys
import argparse
import datetime
import numpy as np
from PIL import Image
import torch
import trimesh
import cv2

from torchvision.utils import save_image

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    
    return rasterizer, renderer
    



def render_vggt(pred_dict, use_point_map: bool = False, max_points: int = 1000000, conf_threshold: float = 20.0, device: str = "cuda", radius: float = 0.003, points_per_pixel: int = 10, bin_size: int = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"output/{timestamp}_max_points_{max_points}_conf_threshold_{conf_threshold}/radius_{radius}_points_per_pixel_{points_per_pixel}_bin_size_{bin_size}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    import ipdb; ipdb.set_trace()

    # predictions_low_res = np.load("output/test_data_soldier_predictions_20250704_111134.npz")
    # extrinsics_cam = predictions_low_res["extrinsic"]
    # intrinsics_cam = predictions_low_res["intrinsic"]
    # intrinsics_cam[:, 0, 0] = intrinsics_cam[:, 0, 0] * 2
    # intrinsics_cam[:, 1, 1] = intrinsics_cam[:, 1, 1] * 2
    # intrinsics_cam[:, 0, 2] = intrinsics_cam[:, 0, 2] * 2
    # intrinsics_cam[:, 1, 2] = intrinsics_cam[:, 1, 2] * 2

    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    depth_fill = np.zeros(depth_map.shape)
    depth_fill_flat = depth_fill.reshape(-1, 1)
    depth_flat = depth_map.reshape(-1, 1)
    
    pts_n, pts_c = points_flat.shape

    if pts_n > max_points:
        # Sort points by confidence
        sorted_indices = np.argsort(conf_flat)[::-1]
        # Select top max_points points
        sorted_indices = sorted_indices[:max_points]
        
        # Use the indices to select the top points
        points_flat = points_flat[sorted_indices]
        colors_flat = colors_flat[sorted_indices]
        conf_flat = conf_flat[sorted_indices]
        depth_flat = depth_flat[sorted_indices]
    

    # Filter points based on confidence threshold
    threshold_val = np.percentile(conf_flat, conf_threshold)
    conf_mask = conf_flat > threshold_val
    points_flat = points_flat[conf_mask]
    colors_flat = colors_flat[conf_mask]
    conf_flat = conf_flat[conf_mask]

    depth_flat = depth_flat[conf_mask]
    depth_flat_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min())
    conf_mask_sorted_indices = sorted_indices[conf_mask]
    depth_fill_flat[conf_mask_sorted_indices] = depth_flat_norm
    depth_fill = depth_fill_flat.reshape(depth_map.shape)

    depth_color, depth_normalized = np_depth_to_colormap(depth_fill[0, :, :, 0])
    cv2.imwrite(os.path.join(save_dir, '%s_vggt_depth_color.png' % (timestamp)), (depth_color).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '%s_vggt_depth.png' % (timestamp)), (depth_normalized*255).astype(np.uint8))


    print("Conf threshold: ", threshold_val)
    print("Total points: ", len(points_flat))

    # Save point cloud for fast visualization
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trimesh.PointCloud(points_flat, colors=colors_flat).export(f"{save_dir}/{timestamp}_points_max_points_{max_points}_conf_{conf_threshold}.ply")

    ## corrdinate transform
    # extrinsics_cam: vggt --> world to camera, vggt is opencv convention
    # transform opencv to pytorch3D convention, x,y,z -> -x, -y, z
    # shape (S, 4, 4) typically, convert w2c into c2w, The rotation of the camera position needs to be executed on the c2w.
    c2w = closed_form_inverse_se3(extrinsics_cam)
    c2w = c2w[:, :3, :]  # For convenience, we store only (3,4) portion

    # c2w_gt = np.load("output/relative_c2w_array_soldier_wood_showpiece-multiview-01-pixel_4a.npz")["relative_c2w_array"][:, :3, :]
    # c2w = c2w_gt
    # K_gt = np.load("output/relative_c2w_array_soldier_wood_showpiece-multiview-01-pixel_4a.npz")["intrinsics"]

    scene_center = np.mean(points_flat, axis=0)
    points_flat_centered = points_flat - scene_center
    c2w[..., -1] -= scene_center
    
    # rotate axis
    opencv_R, T = c2w[:, :3, :3], c2w[:, :3, 3]
    pytorch_three_d_R = np.stack([-opencv_R[:, :, 0], -opencv_R[:, :, 1], opencv_R[:, :, 2]], 2)
    # pytorch_three_d_R = np.stack([-opencv_R[:, 0, :], -opencv_R[:, 1, :], opencv_R[:, 2, :]], 1)
    
    # convert to w2c
    new_c2w = np.concatenate([pytorch_three_d_R, T[:, :, None]], axis=2) # 3*4
    w2c = closed_form_inverse_se3(new_c2w)
    w2c_R, w2c_T = w2c[:, :3, :3], w2c[:, :3, 3]

    # transform intrinsics_cam to 4*4 from 3*3, pinhole camera model
    # K = np.tile(np.eye(4), (S, 1, 1)) # (S, 4, 4)
    # intrinsics_cam[:, 0, 0] = 388.49
    # intrinsics_cam[:, 1, 1] = 388.49
    fx, fy = torch.from_numpy(intrinsics_cam[:, 0, 0]).float(), torch.from_numpy(intrinsics_cam[:, 1, 1]).float()
    ux, uy = torch.from_numpy(intrinsics_cam[:, 0, 2]).float(), torch.from_numpy(intrinsics_cam[:, 1, 2]).float()

    # define camera, pytorch3d is row major, so we need to transpose the R
    w2c_R = torch.from_numpy(w2c_R.transpose(0, 2, 1)).float()
    w2c_T = torch.from_numpy(w2c_T).float()

    image_size = [H, W]

    fcl_screen = torch.stack((fx, fy), dim=1)
    prp_screen = torch.stack((ux, uy), dim=1)
    cameras = PerspectiveCameras(device=device, R=w2c_R, T=w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=[image_size]*S, in_ndc=False,)

    # R, T = look_at_view_transform(0.8, 270, 180)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01)
    # S = 1

    # define rasterizer and renderer
    rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size, radius=radius, points_per_pixel=points_per_pixel, bin_size=bin_size)

    points_flat = torch.from_numpy(points_flat).float().to(device)
    colors_flat = torch.from_numpy(colors_flat/255.0).float().to(device)
    points_flat_centered = torch.from_numpy(points_flat_centered).float().to(device)

    point_cloud = Pointclouds(points=[points_flat_centered]*S, features=[colors_flat]*S)

    new_images, fragments = renderer(point_cloud)

    new_depths = fragments.zbuf[0, :, :, 0].cpu().numpy()
    depth_color, depth_normalized = np_depth_to_colormap(new_depths)

    cv2.imwrite(os.path.join(save_dir, '%s_rendered_depth_color.jpg' % (timestamp)), (depth_color).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, '%s_rendered_depth.jpg' % (timestamp)), (depth_normalized*255).astype(np.uint8))

    for i, new_image in enumerate(new_images):
        new_image = new_image.cpu().numpy()
        new_image = (new_image * 255).astype(np.uint8)
        new_image = Image.fromarray(new_image)
        new_image.save(f"{save_dir}/{timestamp}_new_image_{i}.png")

        images_i = images[i]
        save_image(torch.from_numpy(images_i), f"{save_dir}/{timestamp}_input_image_{i}.png")


    print(f"Rendered images saved to {save_dir}")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--use_point_map", type=bool, default=False)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--conf_threshold", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--points_per_pixel", type=int, default=20)
    parser.add_argument("--bin_size", type=int, default=None)
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser_args()
    predictions = np.load(args.predictions_path)
    render_vggt(predictions, args.use_point_map, args.max_points, args.conf_threshold, device, args.radius, args.points_per_pixel, args.bin_size)

if __name__ == "__main__":
    main()