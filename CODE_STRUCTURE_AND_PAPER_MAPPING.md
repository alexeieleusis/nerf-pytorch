# NeRF Implementation: Code Structure and Paper Mapping

This document maps the PyTorch implementation to the original paper:
**"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"**

## Project Structure Overview

```
nerf-pytorch/
├── run_nerf.py              # Main training and rendering script
├── run_nerf_helpers.py      # Core NeRF components (network, encoding, ray ops)
├── camera_utils.py          # Camera transformation utilities
├── data_utils.py            # Dataset loading utilities (shared across loaders)
├── load_blender.py          # Synthetic dataset loader
├── load_llff.py             # Real-world dataset loader (LLFF format)
├── load_deepvoxels.py       # DeepVoxels dataset loader
├── load_LINEMOD.py          # LINEMOD dataset loader
└── configs/                 # Configuration files for different scenes
```

---

## Code to Paper Section Mapping

### Section 3: Neural Radiance Field Scene Representation

**Paper Concept:** Represent a scene as a continuous 5D function F_Θ: (x, y, z, θ, φ) → (r, g, b, σ)

**Implementation:**

- **File:** `run_nerf_helpers.py`
- **Class:** `NeRF` (lines 95-193)
- **Key Details:**
  - Input: 5D coordinate (3D position + 2D viewing direction)
  - Output: RGB color (3 values) + volume density σ (1 value)
  - Architecture: 8 fully-connected layers, 256 channels per layer
  - Skip connection at layer 5 (concatenates input features)
  - Separate heads for density (view-independent) and color (view-dependent)

```python
# Network architecture from run_nerf_helpers.py:153-193
def forward(self, x):
    # Process position through main MLP
    h = input_pts
    for i in range(D):
        h = pts_linears[i](h)
        h = F.relu(h)
        if i in skips:  # Skip connection at layer 5
            h = torch.cat([input_pts, h], -1)

    # Split into density and color
    alpha = alpha_linear(h)              # σ (view-independent)
    rgb = rgb_linear([h, views])         # RGB (view-dependent)
```

---

### Section 4: Volume Rendering with Radiance Fields

**Paper Concept:** Render color using the volume rendering equation:
```
C(r) = ∫ T(t) · σ(t) · c(t) dt
where T(t) = exp(-∫₀ᵗ σ(s)ds)
```

**Implementation:**

- **File:** `run_nerf.py`
- **Function:** `raw2outputs()` (lines 267-343)
- **Key Details:**
  - Approximates continuous integral using quadrature (stratified sampling)
  - Computes transmittance T(t) using cumulative product
  - Weights each sample by α·T to get final color

```python
# Volume rendering from run_nerf.py:267-343
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    # Convert density to alpha: α = 1 - exp(-σ·δ)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    alpha = raw2alpha(raw[...,3] + noise, dists)

    # Compute transmittance: T_i = ∏ⱼ₌₁ⁱ⁻¹ (1 - αⱼ)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    # Compute expected color: C = Σ wᵢ·cᵢ
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
```

**Ray Generation:**
- **Function:** `get_rays()` in `run_nerf_helpers.py:229-262`
- Generates ray origins and directions for each pixel
- Uses camera intrinsics k and extrinsics c2w

---

### Section 5.1: Positional Encoding

**Paper Concept:** Map input coordinates to higher dimensional space using:
```
γ(p) = (sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
```

**Implementation:**

- **File:** `run_nerf_helpers.py`
- **Class:** `Embedder` (lines 21-58)
- **Function:** `get_embedder()` (lines 61-91)
- **Key Details:**
  - L = 10 frequency bands for position (3D → 63D)
  - L = 4 frequency bands for viewing direction (3D → 27D)
  - Logarithmic frequency sampling

```python
# Positional encoding from run_nerf_helpers.py:21-58
class Embedder:
    def create_embedding_fn(self):
        # Create frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)

        # Apply sin and cos to each frequency
        for freq in freq_bands:
            embed_fns.append(lambda x: torch.sin(x * freq))
            embed_fns.append(lambda x: torch.cos(x * freq))
```

---

### Section 5.2: Hierarchical Volume Sampling

**Paper Concept:** Use a coarse network to guide sampling for a fine network

**Implementation:**

- **File:** `run_nerf.py`
- **Function:** `render_rays()` (lines 377-510)
- **Helper:** `apply_stratified_sampling()` (lines 346-374)
- **Helper:** `sample_pdf()` in `run_nerf_helpers.py:297-370`

**Algorithm:**

1. **Coarse Sampling** (lines 434-466):
   - Stratified sampling: divide ray into n_samples=64 bins
   - Sample randomly within each bin using `apply_stratified_sampling()`
   - Query coarse network at each sample point

```python
# Stratified sampling from run_nerf.py:444-464
# Create stratified samples along the ray
t_vals = torch.linspace(0., 1., steps=n_samples)
z_vals = near * (1.-t_vals) + far * (t_vals)  # Linear spacing

# Add random jitter for stratified sampling (Section 4)
z_vals = apply_stratified_sampling(z_vals, perturb, pytest)

# Compute 3D sample points along rays: r(t) = o + t*d
pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
raw = network_query_fn(pts, viewdirs, network_fn)
```

2. **Fine Sampling** (lines 468-510):
   - Use coarse weights to sample n_importance=128 additional points
   - Inverse transform sampling concentrates samples in high-density regions
   - Combine with coarse samples, query fine network

```python
# Hierarchical sampling from run_nerf.py:468-510
# Use coarse weights as PDF to guide fine sampling
z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], n_importance, det=(perturb==0.), pytest=pytest)
z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
raw = network_query_fn(pts, viewdirs, network_fine)
```

**Inverse Transform Sampling:**
- **Function:** `sample_pdf()` in `run_nerf_helpers.py:297-370`
- Converts weights to PDF, computes CDF
- Samples uniformly from [0,1] and inverts CDF
- Concentrates samples where weights are high

---

### Section 5.3: Implementation Details

**Training Configuration:**

- **File:** `run_nerf.py`
- **Function:** `train()` (lines 1030-1109)
- **Function:** `run_training_loop()` (lines 947-1027)

**Key Hyperparameters:**

| Parameter | Value | Location |
|-----------|-------|----------|
| Batch size (random rays) | 4096 | `config_parser()` line 533 |
| Learning rate | 5e-4 | `config_parser()` line 535 |
| Learning rate decay | 250k steps | `config_parser()` line 537 |
| Coarse samples (n_samples) | 64 | `config_parser()` line 551 |
| Fine samples (n_importance) | 128 | `config_parser()` line 553 |
| Network depth | 8 layers | `config_parser()` line 525 |
| Network width | 256 channels | `config_parser()` line 527 |
| Training iterations | 200k | `run_training_loop()` line 959 |

**Training Loop:**

```python
# Main training loop from run_nerf.py:947-1027
def run_training_loop(args, config: TrainingLoopConfig):
    for i in range(start, n_iters):
        # Sample random ray batch (lines 802-849)
        batch_rays, target_s, ... = get_ray_batch(...)

        # Perform training step (lines 852-889)
        loss, psnr = train_step(batch_rays, target_s, H, W, k, args,
                               render_kwargs_train, optimizer, global_step)

# Training step from run_nerf.py:852-889
def train_step(batch_rays, target_s, h, w, k, args, render_kwargs_train, optimizer, global_step):
    # Render rays using both coarse and fine networks
    rgb, _, _, extras = render(h, w, k, chunk=args.chunk, rays=batch_rays, **render_kwargs_train)

    # Compute photometric loss (MSE)
    img_loss = img2mse(rgb, target_s)  # Fine network loss
    loss = img_loss

    # Add coarse network loss (hierarchical sampling)
    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0  # Total loss = fine_loss + coarse_loss

    # Backprop and optimize
    loss.backward()
    optimizer.step()

    # Exponential learning rate decay
    new_lrate = args.lrate * (0.1 ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    return loss, psnr
```

**Loss Function:**
- Mean Squared Error (L2) between rendered and ground truth RGB
- Both coarse and fine networks trained simultaneously
- Total loss = coarse_loss + fine_loss

---

### Section 5.4: Optimization Details

**Implementation Details:**

- **Optimizer:** Adam (β1=0.9, β2=0.999, weight_decay=0.0)
- **Location:** `create_nerf()` in `run_nerf.py:212`
- **Learning Rate Schedule:**
  - Initial: 5×10⁻⁴
  - Exponential decay by 10× over training
  - `train_step()` in `run_nerf.py:882-886`

**Batching Strategy:**

Two modes available:
1. **Ray Batching** (default): Sample random rays from all training images
   - More memory efficient
   - Better gradient estimates
   - Implemented in `prepare_ray_batching()` (`run_nerf.py:778-800`)
   - Ray sampling in `get_ray_batch()` (`run_nerf.py:802-849`)

2. **Image Batching** (`--no_batching`): Sample rays from one random image at a time
   - Simpler implementation
   - Used for debugging
   - Handled in `get_ray_batch()` (`run_nerf.py:802-849`)

**Optional: Center Cropping** (`--precrop_iters`):
- For first K iterations, only sample rays from image center
- Helps with scenes where object is centered
- Implemented in `get_ray_batch()` (`run_nerf.py:826-846`)

---

## Data Loading and Camera Models

### Blender Synthetic Dataset

- **File:** `load_blender.py`
- **Format:** JSON files with camera poses and RGBA images
- **Key Function:** `load_blender_data()` (lines 12-67)
- **Helper Module:** `data_utils.py` - Common utilities for loading images and poses
  - `load_split_data()` (lines 14-58) - Loads data from train/val/test splits
  - `load_imgs_and_poses_from_meta()` (lines 61-106) - Loads images and camera poses from JSON
- **Details:**
  - Camera poses stored as 4×4 transformation matrices
  - Focal length computed from field of view
  - White background compositing for RGBA images

### Real-World LLFF Dataset

- **File:** `load_llff.py`
- **Format:** `poses_bounds.npy` with camera poses and depth bounds
- **Key Features:**
  - Forward-facing scenes
  - NDC (Normalized Device Coordinates) ray parameterization
  - Sparse view synthesis (typically 20-30 images)

---

## Key Differences from Paper

1. **View-dependent MLP:**
   - Paper suggests multiple layers for view-dependent branch
   - Implementation uses single layer (official TensorFlow version also uses single layer)
   - See comment in `run_nerf_helpers.py:134-140`

2. **Coordinate System:**
   - Uses OpenGL convention (right-handed, Y-up, -Z forward)
   - Camera rays computed using standard pinhole camera model

3. **NDC Rays:**
   - For forward-facing scenes, rays parameterized in normalized device coordinates
   - Helps with unbounded scenes
   - `ndc_rays()` in `run_nerf_helpers.py:276-295`

---

## Usage Examples

### Training on Synthetic Data

```bash
# Train on lego scene
python run_nerf.py --config configs/lego.txt

# Key parameters in config file:
# --datadir: Path to dataset
# --basedir: Output directory for checkpoints
# --N_samples: Number of coarse samples (64)
# --N_importance: Number of fine samples (128)
# --netdepth: Network depth (8)
# --netwidth: Network width (256)
```

### Rendering Novel Views

```bash
# Render test views from trained model
python run_nerf.py --config configs/lego.txt --render_only

# Renders:
# - Test set images
# - Spiral video path
# - Disparity maps
```

---

## Performance Metrics

**From README:**
- Training time: ~4 hours for lego (100k iterations, single 2080 Ti)
- Training time: ~8 hours for fern (200k iterations, single 2080 Ti)
- Speed: 1.3× faster than original TensorFlow implementation
- Results: Numerically matches original implementation

---

## Summary: Complete Pipeline

1. **Input:** Multi-view images with camera poses
   - Loaded by `load_blender.py` or `load_llff.py`

2. **For each training iteration:**
   - Sample 4096 random rays from training images
   - For each ray:
     - a. Stratified sampling → 64 coarse samples
     - b. Query coarse network (with positional encoding)
     - c. Volume rendering → coarse RGB + weights
     - d. Hierarchical sampling → 128 fine samples (guided by weights)
     - e. Query fine network
     - f. Volume rendering → fine RGB
   - Compute loss: MSE(rendered RGB, ground truth)
   - Backprop and optimize both networks

3. **Output:** Trained NeRF models (coarse + fine)
   - Can render novel views
   - Photorealistic quality
   - View-dependent effects (specularities, reflections)

---

## File Reference Quick Guide

| Component | File | Lines |
|-----------|------|-------|
| **Core Training** | | |
| Main training function | `run_nerf.py` | 1030-1109 |
| Training loop | `run_nerf.py` | 947-1027 |
| Training step | `run_nerf.py` | 852-889 |
| Dataset loading dispatcher | `run_nerf.py` | 707-720 |
| Ray batch preparation | `run_nerf.py` | 778-800 |
| Ray batch sampling | `run_nerf.py` | 802-849 |
| **Network & Rendering** | | |
| NeRF MLP architecture | `run_nerf_helpers.py` | 95-227 |
| Create NeRF models | `run_nerf.py` | 183-265 |
| Positional encoding (Embedder) | `run_nerf_helpers.py` | 21-59 |
| Get embedder function | `run_nerf_helpers.py` | 61-92 |
| Volume rendering equation | `run_nerf.py` | 267-343 |
| Render function (main) | `run_nerf.py` | 80-145 |
| Render rays (hierarchical) | `run_nerf.py` | 377-510 |
| **Sampling** | | |
| Stratified sampling helper | `run_nerf.py` | 346-374 |
| Inverse transform sampling | `run_nerf_helpers.py` | 297-370 |
| **Ray Operations** | | |
| Ray generation (torch) | `run_nerf_helpers.py` | 229-264 |
| Ray generation (numpy) | `run_nerf_helpers.py` | 266-274 |
| NDC rays | `run_nerf_helpers.py` | 276-295 |
| **Data Loading** | | |
| Blender synthetic data | `load_blender.py` | 12-67 |
| LLFF real-world data | `load_llff.py` | 62-181 |
| Common data utilities | `data_utils.py` | 14-106 |
| Camera pose utilities | `camera_utils.py` | 24-44 |
| **Configuration** | | |
| Config parser | `run_nerf.py` | 511-622 |
| TrainingLoopConfig dataclass | `run_nerf.py` | 925-944 |

---

## Additional Resources

- **Original Paper:** https://arxiv.org/abs/2003.08934
- **Project Page:** http://www.matthewtancik.com/nerf
- **Official TensorFlow Implementation:** https://github.com/bmild/nerf
- **This PyTorch Implementation:** https://github.com/yenchenlin/nerf-pytorch

---

## Notes for Understanding the Code

1. **Why two networks (coarse and fine)?**
   - Coarse network does initial sampling to find where objects are
   - Fine network focuses computation on relevant regions
   - Significantly improves quality without wasting computation in empty space

2. **Why positional encoding?**
   - MLPs are biased toward learning low-frequency functions
   - High-frequency encoding enables learning fine details
   - Without it, images look blurry

3. **Why stratified sampling?**
   - Prevents aliasing
   - Ensures continuous representation
   - Random jitter within bins during training

4. **Why view-dependent effects?**
   - Real-world materials exhibit specularities, reflections
   - Viewing direction as input enables these effects
   - Density remains view-independent (geometry is fixed)

5. **Memory optimization techniques:**
   - Chunk rays into batches (`--chunk`, `--netchunk`)
   - Process rays and network queries in smaller batches
   - Prevents OOM on consumer GPUs

---

## Code Refactoring History

The codebase has undergone significant refactoring to improve code quality and maintainability:

### New Files Created
1. **`camera_utils.py`** - Camera transformation utilities
   - Extracted camera pose generation functions from data loaders
   - `pose_spherical()` - Generate spherical camera poses for rendering

2. **`data_utils.py`** - Dataset loading utilities
   - Extracted common dataset loading patterns to reduce code duplication
   - `load_split_data()` - Load and combine train/val/test splits
   - `load_imgs_and_poses_from_meta()` - Load images and poses from JSON metadata

### Major Refactoring in `run_nerf.py`

**Training Pipeline Decomposition:**
The monolithic `train()` function was decomposed into smaller, focused functions:
- `train()` - Main entry point for training (setup and coordination)
- `run_training_loop()` - Core training loop iteration
- `train_step()` - Single training step (forward pass, loss, backprop)
- `get_ray_batch()` - Sample ray batches for training
- `prepare_ray_batching()` - Prepare ray batching tensors

**Dataset Loading:**
Dataset loading extracted into dedicated helper functions:
- `load_dataset()` - Dispatcher for different dataset types
- `_load_llff_dataset()` - LLFF dataset loading
- `_load_blender_dataset()` - Blender dataset loading
- `_load_linemod_dataset()` - LINEMOD dataset loading
- `_load_deepvoxels_dataset()` - DeepVoxels dataset loading

**Rendering and Evaluation:**
Rendering and checkpointing extracted into focused functions:
- `handle_render_only()` - Handle render-only mode
- `save_checkpoint()` - Save model checkpoints
- `save_video_renders()` - Render and save video sequences
- `save_testset_renders()` - Render and save test set images

**Sampling:**
Stratified sampling extracted into helper function:
- `apply_stratified_sampling()` - Apply stratified sampling with jitter

**Configuration:**
- `TrainingLoopConfig` dataclass - Encapsulates training loop parameters
- Reduces function parameter count from 19+ to a single config object

**Code Quality Improvements:**
- Renamed variables to follow Python naming conventions (K→k, H→height, W→width, etc.)
- Removed unused variables and parameters
- Removed commented-out code
- Added comprehensive docstrings with mathematical formulas
- Improved cognitive complexity by extracting helper functions
