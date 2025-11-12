import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from data_utils import load_imgs_and_poses_from_meta, load_split_data
from camera_utils import pose_spherical


def load_linemod_data(basedir, half_res=False, testskip=1):
    # Load data from train/val/test splits using shared utility function
    metas, all_imgs, all_poses, counts = load_split_data(
        basedir,
        splits=['train', 'val', 'test'],
        testskip=testskip,
        add_extension=False,
        debug_print=True
    )
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    K = meta['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far


