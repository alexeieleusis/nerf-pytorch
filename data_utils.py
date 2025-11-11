"""
Utility functions for loading dataset images and poses.

This module provides common functionality for loading images and camera poses
from JSON metadata files, used by both Blender and LINEMOD dataset loaders.
"""

import os
import json
import numpy as np
import imageio


def load_split_data(basedir, splits=['train', 'val', 'test'], testskip=1, **kwargs):
    """
    Load data from multiple splits (train, val, test) and combine them.

    This function extracts the common pattern of loading JSON metadata files
    for multiple splits, processing each split with appropriate skip values,
    and concatenating the results.

    Args:
        basedir: Path to dataset directory
        splits: List of split names to load (default: ['train', 'val', 'test'])
        testskip: Load every Nth test/val image (for faster evaluation)
        **kwargs: Additional arguments to pass to load_imgs_and_poses_from_meta

    Returns:
        metas: Dictionary mapping split names to their JSON metadata
        all_imgs: List of image arrays for each split
        all_poses: List of pose arrays for each split
        counts: List of cumulative counts for creating i_split indices
    """
    # Load metadata from JSON files
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # Load images and poses for each split
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        imgs, poses = load_imgs_and_poses_from_meta(
            meta, skip=skip, split_name=s, **kwargs
        )
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    return metas, all_imgs, all_poses, counts


def load_imgs_and_poses_from_meta(meta, basedir=None, skip=1, add_extension=True, debug_print=False, split_name='train'):
    """
    Load images and poses from JSON metadata.

    This helper function extracts the common logic for loading images and camera
    poses from JSON metadata files. It handles reading images, normalizing them
    to [0,1] range, and extracting camera transformation matrices.

    Args:
        meta: Dictionary containing 'frames' list with image metadata
        basedir: Base directory path to prepend to file paths (optional)
        skip: Load every Nth frame (default: 1, load all frames)
        add_extension: If True, append '.png' extension to file paths
        debug_print: If True, print debug information for test frames
        split_name: Name of the split (for debug printing)

    Returns:
        imgs: [N, H, W, C] float32 array of images normalized to [0,1]
        poses: [N, 4, 4] float32 array of camera-to-world transformation matrices
    """
    imgs = []
    poses = []

    for idx, frame in enumerate(meta['frames'][::skip]):
        fname = frame['file_path']

        # Add base directory if provided
        if basedir is not None:
            fname = os.path.join(basedir, fname)

        # Add .png extension if requested
        if add_extension:
            fname = fname + '.png'

        # Debug printing for test frames
        if debug_print and split_name == 'test':
            print(f"{idx}th test frame: {fname}")

        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))

    # Normalize images to [0, 1] range and convert to float32
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    return imgs, poses
