# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of NeRF (Neural Radiance Fields), a method for synthesizing novel views of complex scenes by representing them as continuous volumetric scene functions encoded by neural networks. This implementation reproduces the results from the original paper while running 1.3x faster than the TensorFlow version.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, torchvision, imageio, imageio-ffmpeg, matplotlib, configargparse, tensorboard, tqdm, opencv-python

### Training
```bash
# Train on a specific scene (e.g., lego)
python run_nerf.py --config configs/lego.txt

# Train on other scenes
python run_nerf.py --config configs/{DATASET}.txt
# Available datasets: lego, fern, trex, horns, flower, fortress, chair, drums, ficus, hotdog, ship, etc.
```

### Testing/Rendering
```bash
# Render novel views from trained model
python run_nerf.py --config configs/lego.txt --render_only
```

### Data Download
```bash
# Download example datasets (lego and fern)
bash download_example_data.sh
```

### Testing (Reproducibility)
The repository has a separate branch for reproducibility tests:
```bash
git checkout reproduce
py.test
```

## Code Architecture

### Core Pipeline Overview

NeRF works by training two MLPs (coarse and fine networks) to represent a 3D scene as a continuous volumetric function. The training process:

1. **Sample rays** from input images with known camera poses
2. **Stratified sampling**: Sample 64 points along each ray (coarse)
3. **Positional encoding**: Map 3D coordinates to high-dimensional space
4. **Query coarse network**: Get RGB + density (σ) for each sample point
5. **Volume rendering**: Composite colors using alpha compositing
6. **Hierarchical sampling**: Use coarse weights to sample 128 additional points (fine)
7. **Query fine network**: Get final RGB + density
8. **Loss computation**: MSE between rendered and ground truth RGB
9. **Optimization**: Adam optimizer with exponential learning rate decay

### File Structure and Responsibilities

**Main Training/Rendering:**
- `run_nerf.py` - Main entry point containing training loop (lines 711-872), rendering functions, and configuration parser
  - `render()` - Generates rays and renders images
  - `render_rays()` - Core volume rendering with hierarchical sampling (lines 341-418)
  - `raw2outputs()` - Implements volume rendering equation to convert raw network outputs (RGB+σ) to rendered colors (lines 262-338)
  - `batchify_rays()` - Memory optimization for processing rays in chunks
  - `train()` - Main training loop (lines 534-873)

**Network Architecture:**
- `run_nerf_helpers.py` - Contains all core NeRF components
  - `NeRF` class (lines 95-193) - MLP architecture with skip connections
    - 8 layers × 256 channels with skip connection at layer 5
    - Outputs: volume density σ (view-independent) + RGB color (view-dependent)
  - `Embedder` class (lines 21-58) - Positional encoding γ(p)
    - Maps 3D coords to higher dimensions using sinusoidal functions
    - L=10 freq bands for position (3D → 63D), L=4 for direction (3D → 27D)
  - `get_rays()` (lines 227-261) - Ray generation from camera parameters
  - `sample_pdf()` (lines 295-368) - Inverse transform sampling for hierarchical sampling
  - Helper functions: `img2mse`, `mse2psnr`, `to8b`

**Data Loaders:**
- `load_blender.py` - Loads synthetic datasets (JSON format with camera poses)
  - White background compositing for RGBA images
  - Provides: images, camera poses, focal length, train/val/test splits
- `load_llff.py` - Loads real-world forward-facing scenes (LLFF format)
  - Uses NDC (Normalized Device Coordinates) for unbounded scenes
  - Reads `poses_bounds.npy` with camera poses and depth bounds
- `load_deepvoxels.py` - DeepVoxels dataset loader
- `load_LINEMOD.py` - LINEMOD dataset loader

**Utilities:**
- `camera_utils.py` - Camera-related utility functions
- `data_utils.py` - Data processing utilities

**Configuration:**
- `configs/*.txt` - Per-scene hyperparameter configurations
  - Key params: `N_samples` (64), `N_importance` (128), `N_rand` (batch size)
  - Dataset paths, experiment names, rendering options

### Key Architectural Concepts

**Two-Network Design (Coarse + Fine):**
The coarse network does initial sampling to locate objects, then the fine network focuses computation on relevant regions. Both are trained simultaneously with the same loss function.

**Positional Encoding:**
MLPs bias toward low-frequency functions, so input coordinates are mapped to high-dimensional space using sinusoidal functions. This enables learning fine geometric and appearance details. Without it, rendered images would be blurry.

**Hierarchical Volume Sampling:**
- Coarse: 64 stratified samples along ray (uniform bins with random jitter)
- Fine: 128 additional samples concentrated where coarse network predicts high density
- Uses inverse transform sampling with coarse weights as PDF

**Volume Rendering:**
Implements the continuous volume rendering integral as discrete quadrature:
```
C(r) = Σᵢ Tᵢ · (1 - exp(-σᵢδᵢ)) · cᵢ
where Tᵢ = exp(-Σⱼ₌₁ⁱ⁻¹ σⱼδⱼ)
```
Transmittance T computed via cumulative product, alpha compositing produces final color.

**View-Dependent Effects:**
Viewing direction encoded separately and fed to later layers, enabling specular highlights and reflections while keeping geometry (density) view-independent.

**Memory Optimization:**
- `chunk` parameter: Process rays in smaller batches (default 1024*32)
- `netchunk` parameter: Process network queries in chunks (default 1024*64)
- Essential for running on consumer GPUs

### Important Implementation Details

**Coordinate System:**
- Uses OpenGL convention (right-handed, Y-up, -Z forward)
- Camera transformation: c2w (camera-to-world) 3×4 matrix

**NDC Rays (for LLFF forward-facing scenes):**
Rays parameterized in normalized device coordinates to handle unbounded scenes (see `run_nerf_helpers.py:246-262`).

**Training Hyperparameters:**
- Batch size: 4096 random rays (or 1024 in configs)
- Learning rate: 5e-4 with exponential decay to 5e-5 over training
- Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-7)
- Training iterations: 100k-200k depending on scene complexity
- Loss: MSE between rendered and ground truth RGB for both networks

**Batching Modes:**
- Default (ray batching): Sample random rays from all training images
- Alternative (`no_batching=True`): Process one full image at a time
- Optional pre-cropping (`precrop_iters`): Initially sample rays from image center

**View-Dependent MLP:**
The implementation follows the official TensorFlow code using a single layer for the view-dependent branch, not multiple layers as suggested in the paper diagram (see comment in `run_nerf_helpers.py:136-142`).

## Dataset Structure

Expected directory layout:
```
data/
├── nerf_synthetic/          # Blender synthetic datasets
│   ├── lego/
│   ├── ship/
│   └── ...
├── nerf_llff_data/          # Real-world LLFF datasets
│   ├── fern/
│   ├── flower/
│   └── ...
```

Pre-trained models go in:
```
logs/
├── lego_test/
├── fern_test/
└── ...
```

## Configuration File Parameters

Key parameters in `configs/*.txt`:
- `expname` - Experiment name for logging
- `basedir` - Output directory (default: `./logs`)
- `datadir` - Path to dataset
- `dataset_type` - `blender`, `llff`, `deepvoxels`, or `linemod`
- `N_samples` - Number of coarse samples per ray (typically 64)
- `N_importance` - Number of fine samples per ray (typically 64-128)
- `N_rand` - Batch size (rays per gradient step, typically 1024)
- `use_viewdirs` - Enable view-dependent effects (True for best quality)
- `white_bkgd` - Use white background (True for synthetic data)
- `half_res` - Use half resolution images for faster training
- `factor` - Downsample factor for LLFF data
- `llffhold` - Use every Nth image as test set for LLFF
- `lrate_decay` - Learning rate decay iterations
- `precrop_iters` - Number of iterations to train on center crops

## Training Performance

- Lego (100k iterations): ~4 hours on single 2080 Ti
- Fern (200k iterations): ~8 hours on single 2080 Ti
- Results numerically match official TensorFlow implementation
- 1.3× speedup compared to TensorFlow version

## Additional Resources

- Original paper: https://arxiv.org/abs/2003.08934
- Project page: http://www.matthewtancik.com/nerf
- Official TensorFlow implementation: https://github.com/bmild/nerf
- Full code-to-paper mapping: See `CODE_STRUCTURE_AND_PAPER_MAPPING.md`
