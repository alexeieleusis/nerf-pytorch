import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from data_utils import load_imgs_and_poses_from_meta, load_split_data
from camera_utils import pose_spherical


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    Load synthetic Blender dataset.

    The Blender dataset consists of synthetic scenes with known camera poses
    and perfect ground truth. Each scene includes:
    - RGBA images (with alpha channel for compositing)
    - Camera transformation matrices
    - Camera intrinsics (field of view)

    Args:
        basedir: Path to dataset directory
        half_res: If True, downsample images to 400x400
        testskip: Load every Nth test/val image (for faster evaluation)

    Returns:
        imgs: [N, H, W, 4] RGBA images
        poses: [N, 4, 4] camera-to-world transformation matrices
        render_poses: Camera poses for novel view synthesis
        hwf: [H, W, focal] image dimensions and focal length
        i_split: Indices for train/val/test splits
    """
    # Load data from train/val/test splits using shared utility function
    metas, all_imgs, all_poses, counts = load_split_data(
        basedir,
        splits=['train', 'val', 'test'],
        testskip=testskip,
        add_extension=True,
        debug_print=False
    )
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res


    return imgs, poses, render_poses, [H, W, focal], i_split


