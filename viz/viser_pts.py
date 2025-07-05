# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from PIL import Image
from einops import rearrange



try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def viser_wrapper(
    points_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    vis_camera: bool = False,
    update_throttle: float = 0.2,  # Throttle update frequency in seconds
    point_size: float = 0.0001,  # Default point size
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        points_dict (dict):
            {
                "images": (S, 3, H, W) or (N, 3) - Input images,
                "world_points": (S, H, W, 3) or (N, 3) - World points,
                "world_points_conf": (S, H, W) or (N, 3) - World points confidence,
                "depth": (S, H, W, 1) or (N, 3) - Depth map,
                "depth_conf": (S, H, W) or (N, 3) - Depth confidence,
                "extrinsic": (S, 3, 4)  - Extrinsic matrix,
                "intrinsic": (S, 3, 3)  - Intrinsic matrix,
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        update_throttle (float): Minimum time between point cloud updates in seconds.
        point_size (float): Initial point size for visualization.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = points_dict["images"]  # (S, 3, H, W)
    world_points_map = points_dict["world_points"]  # (S, H, W, 3)
    conf_map = points_dict["world_points_conf"]  # (S, H, W)
    

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        depth_map = points_dict["depth"]  # (S, H, W, 1)
        depth_conf = points_dict["depth_conf"]  # (S, H, W)
        extrinsics_cam = points_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = points_dict["intrinsic"]  # (S, 3, 3)
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map


    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    is_shw_sequence = images.ndim == 4
    if is_shw_sequence:
        colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
        S, H, W, _ = world_points.shape
        # Flatten
        points = world_points.reshape(-1, 3)
        colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = conf.reshape(-1)

        # Store frame indices so we can filter by frame
        frame_indices = np.repeat(np.arange(S), H * W)
    else:
        colors_flat = images
        # images = rearrange(images, '(h w) c -> h w c', h = 392, w = 518)
        # Image.fromarray(images.astype(np.uint8)).save('test.png')
        points = world_points
        conf_flat = conf.reshape(-1)

    # import ipdb; ipdb.set_trace()
    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center


    if vis_camera:
        cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically, convert w2c into c2w
        # For convenience, we store only (3,4) portion
        cam_to_world = cam_to_world_mat[:, :3, :]
        # recenter
        cam_to_world[..., -1] -= scene_center

        # Build the viser GUI
        gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

        gui_frame_selector = server.gui.add_dropdown(
            "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
        )


    # import ipdb; ipdb.set_trace()
    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=1, initial_value=init_conf_threshold
    )

    # Add point size slider
    gui_point_size = server.gui.add_slider(
        "Point Size", min=0.00001, max=0.01, step=0.0001, initial_value=point_size
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask][:, :3],
        point_size=point_size,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility       
    if vis_camera:
        frames: List[viser.FrameHandle] = []
        frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    # Variables for throttling updates
    last_update_time = 0
    pending_update = False
    current_percentage = init_conf_threshold

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        nonlocal last_update_time, pending_update
        
        # Check if we need to throttle the update
        current_time = time.time()
        if current_time - last_update_time < update_throttle:
            # If an update is already pending, just return
            if pending_update:
                return
                
            # Schedule an update for later
            pending_update = True
            
            def delayed_update():
                nonlocal pending_update, last_update_time
                time.sleep(max(0, update_throttle - (time.time() - last_update_time)))
                update_point_cloud_impl()
                pending_update = False
                last_update_time = time.time()
                
            threading.Thread(target=delayed_update, daemon=True).start()
            return
            
        # Perform the actual update
        update_point_cloud_impl()
        last_update_time = current_time
        pending_update = False
        
    def update_point_cloud_impl() -> None:
        """Actual implementation of point cloud update"""
        nonlocal current_percentage
        # Get the current percentage from the slider
        current_percentage = gui_points_conf.value
        
        # Here we compute the threshold value based on the current percentage
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)


        if vis_camera:
            if gui_frame_selector.value == "All":
                frame_mask = np.ones_like(conf_mask, dtype=bool)
            else:
                selected_idx = int(gui_frame_selector.value)
                frame_mask = frame_indices == selected_idx
        else:
            frame_mask = np.ones_like(conf_mask, dtype=bool)

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask][:, :3]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_point_size.on_update
    def _(_) -> None:
        """Update point size."""
        point_cloud.point_size = gui_point_size.value

    if vis_camera:
        @gui_frame_selector.on_update
        def _(_) -> None:
            update_point_cloud()

        @gui_show_frames.on_update
        def _(_) -> None:
            """Toggle visibility of camera frames and frustums."""
            for f in frames:
                f.visible = gui_show_frames.value
            for fr in frustums:
                fr.visible = gui_show_frames.value


    # Add the camera frames to the scene
    if vis_camera:
        visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--pts_path", type=str, default="third_party/pytorch3d/docs/tutorials/data/PittsburghBridge/pointcloud.npz", help="Path to point clouds")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=12345, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=15.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--update_throttle", type=float, default=0.2, help="Minimum time between point cloud updates in seconds")
parser.add_argument("--point_size", type=float, default=0.002, help="Point size in visualization")


def main():
    """
    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --update_throttle: Minimum time between point cloud updates in seconds
    --point_size: Point size in visualization
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    points_dict = {}

    point_clouds = np.load(args.pts_path)

    if 'rgb' in point_clouds:
        rgb = point_clouds['rgb']
    else:
        image_names = glob.glob(os.path.join(args.image_folder, "*"))
        rgb = load_and_preprocess_images(image_names)

    points_dict['images'] = rgb
    points_dict['world_points'] = point_clouds['verts']

    if 'conf' in point_clouds:
        points_dict['world_points_conf'] = point_clouds['conf']
    else:
        points_dict['world_points_conf'] = np.ones_like(points_dict['world_points'][:,0])

    if 'depth' in point_clouds:
        points_dict['depth'] = point_clouds['depth']
    else:
        points_dict['depth'] = np.ones_like(points_dict['world_points'])

    if 'extrinsic' in point_clouds:
        points_dict['extrinsic'] = point_clouds['extrinsic']
    else:
        points_dict['extrinsic'] = np.eye(3, 4)

    if 'intrinsic' in point_clouds:
        points_dict['intrinsic'] = point_clouds['intrinsic']
    else:
        points_dict['intrinsic'] = np.eye(3, 3)


    import ipdb; ipdb.set_trace()
    viser_server = viser_wrapper(
        points_dict,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        update_throttle=args.update_throttle,
        point_size=args.point_size,
        vis_camera='extrinsic' in point_clouds,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
