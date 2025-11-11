# NeRF Implementation: Code Structure and Paper Mapping

This document maps the PyTorch implementation to the original paper:
**"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"**

## Project Structure Overview

```
nerf-pytorch/
├── run_nerf.py              # Main training and rendering script
├── run_nerf_helpers.py      # Core NeRF components (network, encoding, ray ops)
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
- **Function:** `raw2outputs()` (lines 262-338)
- **Key Details:**
  - Approximates continuous integral using quadrature (stratified sampling)
  - Computes transmittance T(t) using cumulative product
  - Weights each sample by α·T to get final color

```python
# Volume rendering from run_nerf.py:262-338
def raw2outputs(raw, z_vals, rays_d, ...):
    # Convert density to alpha: α = 1 - exp(-σ·δ)
    alpha = 1. - torch.exp(-F.relu(raw[...,3]) * dists)

    # Compute transmittance: T_i = ∏ⱼ₌₁ⁱ⁻¹ (1 - αⱼ)
    weights = alpha * torch.cumprod(1.-alpha + 1e-10, -1)

    # Compute expected color: C = Σ wᵢ·cᵢ
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
```

**Ray Generation:**
- **Function:** `get_rays()` in `run_nerf_helpers.py:227-261`
- Generates ray origins and directions for each pixel
- Uses camera intrinsics K and extrinsics c2w

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
- **Function:** `render_rays()` (lines 341-418)
- **Helper:** `sample_pdf()` in `run_nerf_helpers.py:295-368`

**Algorithm:**

1. **Coarse Sampling** (lines 400-446):
   - Stratified sampling: divide ray into N_samples=64 bins
   - Sample randomly within each bin
   - Query coarse network at each sample point

```python
# Stratified sampling from run_nerf.py:408-446
z_vals = near * (1.-t_vals) + far * t_vals  # Linear spacing
if perturb > 0.:
    # Add jitter within each bin
    z_vals = lower + (upper - lower) * torch.rand(...)
pts = rays_o + rays_d * z_vals  # 3D points
raw = network_query_fn(pts, viewdirs, network_fn)
```

2. **Fine Sampling** (lines 448-474):
   - Use coarse weights to sample N_importance=128 additional points
   - Inverse transform sampling concentrates samples in high-density regions
   - Combine with coarse samples, query fine network

```python
# Hierarchical sampling from run_nerf.py:448-474
# Use coarse weights as PDF to guide fine sampling
z_samples = sample_pdf(z_vals_mid, weights, N_importance)
z_vals = torch.sort(torch.cat([z_vals, z_samples], -1))
raw = network_query_fn(pts, viewdirs, network_fine)
```

**Inverse Transform Sampling:**
- **Function:** `sample_pdf()` in `run_nerf_helpers.py:295-368`
- Converts weights to PDF, computes CDF
- Samples uniformly from [0,1] and inverts CDF
- Concentrates samples where weights are high

---

### Section 5.3: Implementation Details

**Training Configuration:**

- **File:** `run_nerf.py`
- **Function:** `train()` (lines 534-873)

**Key Hyperparameters:**

| Parameter | Value | Location |
|-----------|-------|----------|
| Batch size (random rays) | 4096 | `config_parser()` line 443 |
| Learning rate | 5e-4 | `config_parser()` line 445 |
| Learning rate decay | 250k steps | `config_parser()` line 448 |
| Coarse samples (N_c) | 64 | `config_parser()` line 461 |
| Fine samples (N_f) | 128 | `config_parser()` line 463 |
| Network depth | 8 layers | `config_parser()` line 435 |
| Network width | 256 channels | `config_parser()` line 437 |
| Training iterations | 200k | `train()` line 701 |

**Training Loop:**

```python
# Training loop from run_nerf.py:711-872
for i in range(200000):
    # Sample random rays from training images
    batch_rays, target_s = get_random_rays(...)

    # Render rays
    rgb, disp, acc, extras = render(batch_rays, ...)

    # Compute photometric loss
    loss = img2mse(rgb, target_s)  # Fine network loss
    if 'rgb0' in extras:
        loss += img2mse(extras['rgb0'], target_s)  # Coarse network loss

    # Optimize
    loss.backward()
    optimizer.step()

    # Exponential learning rate decay
    new_lrate = lrate * (0.1 ** (global_step / decay_steps))
```

**Loss Function:**
- Mean Squared Error (L2) between rendered and ground truth RGB
- Both coarse and fine networks trained simultaneously
- Total loss = coarse_loss + fine_loss

---

### Section 5.4: Optimization Details

**Implementation Details:**

- **Optimizer:** Adam (β1=0.9, β2=0.999)
- **Location:** `run_nerf.py:207`
- **Learning Rate Schedule:**
  - Initial: 5×10⁻⁴
  - Exponential decay by 10× over training
  - `run_nerf.py:860-864`

**Batching Strategy:**

Two modes available:
1. **Ray Batching** (default): Sample 4096 random rays from all training images
   - More memory efficient
   - Better gradient estimates
   - `run_nerf.py:677-698`

2. **Image Batching** (`--no_batching`): Sample rays from one random image
   - Simpler implementation
   - Used for debugging
   - `run_nerf.py:728-757`

**Optional: Center Cropping** (`--precrop_iters`):
- For first K iterations, only sample rays from image center
- Helps with scenes where object is centered
- `run_nerf.py:738-747`

---

## Data Loading and Camera Models

### Blender Synthetic Dataset

- **File:** `load_blender.py`
- **Format:** JSON files with camera poses and RGBA images
- **Key Function:** `load_blender_data()` (lines 52-89)
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
   - `run_nerf_helpers.py:246-262`

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
| Main training loop | `run_nerf.py` | 711-872 |
| NeRF MLP architecture | `run_nerf_helpers.py` | 95-193 |
| Positional encoding | `run_nerf_helpers.py` | 21-91 |
| Volume rendering | `run_nerf.py` | 262-338 |
| Stratified sampling | `run_nerf.py` | 400-446 |
| Hierarchical sampling | `run_nerf.py` | 448-474 |
| Inverse transform sampling | `run_nerf_helpers.py` | 295-368 |
| Ray generation | `run_nerf_helpers.py` | 227-261 |
| Data loading (Blender) | `load_blender.py` | 52-89 |
| Data loading (LLFF) | `load_llff.py` | 62-181 |
| Config parser | `run_nerf.py` | 421-531 |

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
