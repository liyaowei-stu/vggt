import torch
import numpy as np


def convert_se3_to_homogeneous(se3):
    """
    Args:
        se3: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)

    convert se3 to homogeneous matrix

    Returns:
        homogeneous: similar to se3
    """
    input_dim = len(se3.shape)
    if len(se3.shape) == 2:
        se3 = se3[None][None]
    if len(se3.shape) == 3:
        se3 = se3[None]
    
    homogeneous = torch.eye(4).to(se3.device)

    homogeneous = homogeneous[None, None].repeat(se3.shape[0], se3.shape[1], 1, 1)
    homogeneous[:, :, :3, :3] = se3[:, :, :3, :3]
    homogeneous[:, :, :3, 3] = se3[:, :, :3, 3]

    while len(homogeneous.shape) != input_dim:
        homogeneous = homogeneous.squeeze(0)
    
    return homogeneous


def convert_homogeneous_to_se3(homogeneous):
    """
    Args:
        homogeneous: (b, 4, 4) or (4, 4) or (b, s, 4, 4) or (s, 4, 4)
    """
    if len(homogeneous.shape) == 3:
        homogeneous = homogeneous[:, :3, :]
    elif len(homogeneous.shape) == 4:
        homogeneous = homogeneous[:, :, :3, :]
    elif len(homogeneous.shape) == 2:
        homogeneous = homogeneous[:3, :]
    else:
        raise ValueError(f"relative_w2c has invalid shape: {homogeneous.shape}")
    return homogeneous


def convert_camera_pose_to_relative(extrinsic, anchor_extrinsic):
    """
    Args:
        extrinsic: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)
        anchor_extrinsic: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)

    Returns:
        relative_w2c: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)
    """
    extrinsic = convert_se3_to_homogeneous(extrinsic).to(torch.float64)
    anchor_extrinsic = convert_se3_to_homogeneous(anchor_extrinsic).to(torch.float64)

    # assert len(extrinsic.shape) == len(anchor_extrinsic.shape), "extrinsic and anchor_extrinsic must have the same number of dimensions"

    c2w = torch.linalg.inv(extrinsic)
    
    relative_c2w = anchor_extrinsic @ c2w

    relative_w2c = torch.linalg.inv(relative_c2w)

    relative_w2c = convert_homogeneous_to_se3(relative_w2c)

    relative_w2c = relative_w2c.to(torch.float32)

    return relative_w2c


def calculate_extrinsic_correction(extrinsics_1, extrinsics_2):
    """
    Calculate the correction matrix between extrinsics_1 and extrinsics_2
    The correction matrix transforms extrinsics_1 to match extrinsics_2

    Args:
        extrinsics_1: (b, s, 3, 4)
        extrinsics_2: (b, s, 3, 4)

    Returns:
        correction_matrix: (b, s, 3, 4)
    """
    extrinsics_1 = convert_se3_to_homogeneous(extrinsics_1).to(torch.float64)
    extrinsics_2 = convert_se3_to_homogeneous(extrinsics_2).to(torch.float64)

    correction_matrix = torch.matmul(extrinsics_2, torch.linalg.inv(extrinsics_1))

    correction_matrix = correction_matrix[:, :, :3, :].to(torch.float32)

    correction_matrix = correction_matrix.mean(dim=1)

    correction_matrix = correction_matrix.unsqueeze(1)

    return correction_matrix


def excute_extrinsic_correction(extrinsics, correction_matrix):
    """
    Execute the correction matrix on the extrinsics

    Args:
        extrinsics: (b, s, 3, 4)
        correction_matrix: (b, s, 3, 4)

    Returns:
        corrected_extrinsics: (b, s, 3, 4)
    """

    extrinsics = convert_se3_to_homogeneous(extrinsics).to(torch.float64)
    correction_matrix = convert_se3_to_homogeneous(correction_matrix).to(torch.float64)

    corrected_extrinsics = torch.matmul(correction_matrix, extrinsics)

    corrected_extrinsics = convert_homogeneous_to_se3(corrected_extrinsics)

    corrected_extrinsics = corrected_extrinsics.to(torch.float32)

    return corrected_extrinsics


def calculate_intrinsic_correction(intrinsic_1, intrinsic_2):
    """
    Calculate the scale between two intrinsic matrices.
    Mainly focuses on the focal length scale change.
    The scale transforms intrinsic_1 to match intrinsic_2
    
    Args:
        intrinsic_1: (B, 3, 3) or (B, S, 3, 3)
        intrinsic_2: (B, 3, 3) or (B, S, 3, 3)
        
    Returns:
        correction_scale: (B,) or (B, S) - scale factors for focal length
    """
    # Extract focal lengths (fx, fy)
    if len(intrinsic_1.shape) == 3:  # (B, 3, 3)
        fx1, fy1 = intrinsic_1[:, 0, 0], intrinsic_1[:, 1, 1]
        fx2, fy2 = intrinsic_2[:, 0, 0], intrinsic_2[:, 1, 1]
    else:  # (B, S, 3, 3)
        fx1, fy1 = intrinsic_1[:, :, 0, 0], intrinsic_1[:, :, 1, 1]
        fx2, fy2 = intrinsic_2[:, :, 0, 0], intrinsic_2[:, :, 1, 1]
    
    # Calculate scale factors
    scale_x = fx2 / fx1
    scale_y = fy2 / fy1
    
    # Average the scale factors (could also use just one if they should be identical)
    correction_scale = (scale_x + scale_y) / 2.0

    scale_x = scale_x.mean()
    scale_y = scale_y.mean()
    correction_scale = correction_scale.mean()
    
    return scale_x, scale_y, correction_scale


def excute_intrinsic_correction(intrinsic, scale_x, scale_y, intrinsic_scale=None):
    """
    Execute the scale on the intrinsic.
    Args:
        intrinsic: (B, 3, 3) or (B, S, 3, 3)
        scale_x: (B,) or (B, S)
        scale_y: (B,) or (B, S)

    Returns:
        corrected_intrinsic: (B, 3, 3) or (B, S, 3, 3)
    """
    if intrinsic_scale is not None:
        scale_x = intrinsic_scale
        scale_y = intrinsic_scale

    intrinsic = intrinsic.clone()
    if len(intrinsic.shape) == 3:
        intrinsic[:, 0, 0] *= scale_x
        intrinsic[:, 1, 1] *= scale_y
    else:
        intrinsic[:, :, 0, 0] *= scale_x
        intrinsic[:, :, 1, 1] *= scale_y

    return intrinsic