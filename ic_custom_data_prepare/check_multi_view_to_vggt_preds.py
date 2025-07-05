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



@torch.no_grad()
def main(
    image_path, 
    model, 
    use_point_map: bool = False, 
    max_points: int = 1000000, 
    conf_threshold: float = 20.0
    ):

    import ipdb; ipdb.set_trace()

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
        np.savez(f"output/test_data_soldier_predictions_{timestamp}_all.npz", **predictions)




def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B")    
    return parser.parse_args()



if __name__ == "__main__":
    args = parser_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    image_path = sorted(glob.glob(os.path.join(args.image_path, "*")))

    model = VGGT.from_pretrained(args.model).to(device)

    main(image_path, model)
