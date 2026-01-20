"""
NeRF: Neural Radiance Fields for View Synthesis
================================================

This module implements the training and rendering pipeline for NeRF as described in:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
by Mildenhall et al., ECCV 2020

Key Concepts:
- Represents scenes as continuous 5D functions: (x, y, z, θ, φ) → (R, G, B, σ)
- Uses positional encoding to capture high-frequency scene details
- Employs volume rendering to synthesize novel views
- Hierarchical sampling strategy for efficient computation

Main Components:
- Volume Rendering: Classical rendering equation (Section 4)
- Positional Encoding: Maps coordinates to higher dimensions (Section 5.1)
- Hierarchical Sampling: Two-stage coarse-to-fine sampling (Section 5.2)
- Photometric Loss: MSE between rendered and ground truth images

Usage:
    python run_nerf.py --config configs/lego.txt  # Train on synthetic lego scene
    python run_nerf.py --config configs/fern.txt  # Train on real forward-facing scene
"""
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from load_blender import load_blender_data
from load_deepvoxels import load_dv_data
from load_LINEMOD import load_linemod_dataset
from load_llff import load_llff_data
from run_nerf_helpers import NeRF, get_embedder, get_rays, get_rays_np, img2mse, mse2psnr, ndc_rays, sample_pdf, to8b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches for memory efficiency.

    This is a wrapper function that splits large inputs into smaller chunks to prevent
    out-of-memory errors when processing through neural networks. This is particularly
    important for NeRF because:
    - Networks may receive thousands of query points simultaneously
    - Full-resolution rendering can require millions of network evaluations
    - GPU memory is limited

    Args:
        fn: Function to apply (typically a neural network forward pass)
        chunk: Maximum number of inputs to process at once. If None, process all at once.

    Returns:
        ret: Wrapped function that processes inputs in chunks and concatenates results
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Prepares inputs with positional encoding and applies network in batches.

    This function:
    1. Flattens 3D position inputs
    2. Applies positional encoding to positions (gamma function from Section 5.1)
    3. Optionally applies positional encoding to viewing directions
    4. Concatenates encoded position and direction
    5. Queries the network in chunks for memory efficiency
    6. Reshapes output back to original batch structure

    Args:
        inputs: [N_rays, N_samples, 3] 3D sample positions along rays
        viewdirs: [N_rays, 3] Viewing directions for each ray, or None
        fn: Neural network function (NeRF model)
        embed_fn: Positional encoding function for 3D positions
        embeddirs_fn: Positional encoding function for viewing directions
        netchunk: Maximum number of points to send through network at once

    Returns:
        outputs: [N_rays, N_samples, 4] Network predictions (RGB + sigma)
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, [*list(inputs.shape[:-1]), outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """
    Render rays in smaller minibatches to avoid out-of-memory errors.

    During rendering, we may need to process thousands of rays simultaneously. Each ray
    requires sampling multiple points (N_samples + N_importance), and each point requires
    a network evaluation. This function splits the rays into smaller batches to manage
    memory usage while maintaining identical results to processing all rays at once.

    Args:
        rays_flat: [N_rays, ...] Batch of rays to render
        chunk: Maximum number of rays to process simultaneously (default: 32768)
        **kwargs: Additional arguments passed to render_rays()

    Returns:
        all_ret: Dictionary containing rendering outputs (rgb_map, disp_map, acc_map, etc.)
                 concatenated across all batches
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
    height,
    width,
    focal,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    verbose=False,
    **kwargs,
):
    """
    Render rays to generate RGB image, depth map, and opacity.

    This is the main rendering interface that orchestrates the entire volume rendering
    pipeline. It can either render a full image from a camera pose or render a batch
    of pre-generated rays.

    Args:
        height: Image height in pixels (H in paper notation)
        width: Image width in pixels (W in paper notation)
        focal: Focal length of pinhole camera (can be intrinsic matrix K)
        chunk: Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: Array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: Array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: If True, represent ray origin, direction in NDC coordinates (for forward-facing scenes).
        near: Nearest distance for a ray (float or array of shape [batch_size])
        far: Farthest distance for a ray (float or array of shape [batch_size])
        use_viewdirs: If True, use viewing direction for view-dependent effects (Section 5.1)
        c2w_staticcam: Array of shape [3, 4]. If not None, use this transformation matrix for
            camera while using other c2w argument for viewing directions.
        verbose: If True, print debugging information
        **kwargs: Additional arguments passed to render_rays()

    Returns:
        rgb_map: [batch_size, 3] Predicted RGB values for rays
        disp_map: [batch_size] Disparity map (inverse of depth)
        acc_map: [batch_size] Accumulated opacity (alpha) along each ray
        extras: Dictionary with everything returned by render_rays() (includes coarse outputs if hierarchical)
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(height, width, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(height, width, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(height, width, focal[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = [*list(sh[:-1]), *list(all_ret[k].shape[1:])]
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return [*ret_list, ret_dict]


def render_path(render_poses, hwf, k, chunk, render_kwargs, savedir=None, render_factor=0):
    """
    Render images from a sequence of camera poses (for creating videos).

    This function is used to:
    1. Render test set images for quantitative evaluation
    2. Generate novel view synthesis videos (spiral/sphere paths)
    3. Visualize the learned scene representation

    Args:
        render_poses: [N, 3, 4] Sequence of camera-to-world transformation matrices
        hwf: [3] Height, width, focal length tuple
        k: [3, 3] Camera intrinsic matrix
        chunk: Maximum number of rays to process simultaneously
        render_kwargs: Dictionary of rendering parameters (networks, sampling settings, etc.)
        savedir: Directory to save rendered images (if None, don't save individual frames)
        render_factor: Downsampling factor (0 = full resolution, 2 = half, 4 = quarter, etc.)

    Returns:
        rgbs: [N, H, W, 3] Rendered RGB images
        disps: [N, H, W] Disparity maps
    """

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, _acc, _ = render(H, W, k, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, f"{i:03d}.png")
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
    Instantiate NeRF's MLP model and optimizer.

    This function sets up the complete NeRF architecture:
    1. Creates positional encoding functions for position and viewing direction
    2. Instantiates coarse network
    3. Instantiates fine network (if hierarchical sampling is enabled)
    4. Sets up Adam optimizer
    5. Loads checkpoint if available

    The two-network setup (coarse + fine) implements hierarchical volume sampling
    from Section 5.2 of the paper.

    Args:
        args: Parsed command-line arguments containing hyperparameters

    Returns:
        render_kwargs_train: Dictionary of parameters for training rendering
        render_kwargs_test: Dictionary of parameters for test rendering
        start: Starting iteration number (for resuming training)
        grad_vars: List of parameters to optimize
        optimizer: Adam optimizer instance
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        depth=args.netdepth,
        width=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            depth=args.netdepth_fine,
            width=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), weight_decay=0)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "n_importance": args.N_importance,
        "network_fine": model_fine,
        "n_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """
    Transforms model's predictions to semantically meaningful values using volume rendering.

    This implements the classical volume rendering integral described in Section 4:
    C(r) = ∫ T(t) · sigma(t) · c(t) dt

    where:
    - C(r) is the expected color along ray r
    - T(t) = exp(-∫₀ᵗ sigma(s)ds) is the transmittance (probability ray travels to t without hitting anything)
    - sigma(t) is the volume density at point t
    - c(t) is the RGB color at point t

    The continuous integral is approximated using quadrature (numerical integration)
    with stratified sampling along the ray.

    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model (RGB + sigma).
        z_vals: [num_rays, num_samples along ray]. Sample distances along each ray.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map (inverse depth).
        acc_map: [num_rays]. Accumulated opacity (sum of weights).
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Expected distance to surface.
    """
    # Function to convert raw density to alpha (opacity) using exponential
    # Formula: alpha = 1 - exp(-sigma·delta), where delta is the distance between samples
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]
    # Last distance is set to infinity to handle ray endpoints

    # Scale distances by ray direction norm to get actual Euclidean distances
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Apply sigmoid to get RGB in [0, 1]
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    # Optional: Add noise to density predictions during training for regularization
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*raw[..., 3].shape) * raw_noise_std
            noise = torch.tensor(noise, device=raw.device)

    # Compute alpha (opacity) from density
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # Compute transmittance T(t) = exp(-∫₀ᵗ sigma(s)ds)
    # Using the cumulative product: T_i = ∏ⱼ₌₁ⁱ⁻¹ (1 - alpha_j)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha
        * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], -1), -1)[
            :, :-1
        ]
    )
    # weights[i] = T_i · alpha_i represents the probability that ray terminates at sample i

    # Compute expected color using quadrature: C = Σ wᵢ·cᵢ
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # Compute expected depth: E[t] = Σ wᵢ·tᵢ
    depth_map = torch.sum(weights * z_vals, -1)
    # Compute disparity (inverse depth)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # Accumulated opacity: how much "stuff" is along the ray
    acc_map = torch.sum(weights, -1)

    # If white background, composite the predicted RGB with white background
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def _create_stratified_samples(rays_o, near, far, n_samples, lindisp, perturb, pytest):
    """
    Create stratified samples along rays (Section 4 - Stratified Sampling).

    Stratified sampling divides each ray into N_samples evenly-spaced bins and
    samples one point randomly within each bin. This approach:
    1. Ensures continuous representation of the scene
    2. Prevents aliasing from regular sampling
    3. Enables optimization of the continuous volumetric representation

    Args:
        rays_o: [N_rays, 3] Ray origins
        near: Near plane distance for sampling
        far: Far plane distance for sampling
        n_samples: Number of samples per ray (typically 64 for coarse network)
        lindisp: If True, sample linearly in disparity (1/depth) rather than depth
        perturb: If > 0, add random jitter within bins (training). If 0, deterministic (testing)
        pytest: If True, use fixed random seed for testing

    Returns:
        z_vals: [N_rays, N_samples] Sample depths along each ray
    """
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=rays_o.device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([rays_o.shape[0], n_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*z_vals.shape)
            t_rand = torch.tensor(t_rand, device=z_vals.device)

        z_vals = lower + (upper - lower) * t_rand

    return z_vals


def _perform_hierarchical_sampling(z_vals, weights, n_importance, perturb, pytest):
    """
    Perform hierarchical sampling using coarse network weights (Section 5.2).

    This implements the two-stage sampling strategy:
    1. Coarse network identifies where the scene content is (via density predictions)
    2. Fine network focuses samples on relevant regions using importance sampling

    The coarse network's weights form a probability distribution along each ray,
    indicating where volume density (and thus scene content) is likely to be.
    We use inverse transform sampling to draw additional samples from this distribution,
    concentrating computation on regions that matter.

    This approach significantly improves quality without wasting samples on empty space.

    Args:
        z_vals: [N_rays, N_samples] Sample depths from coarse network
        weights: [N_rays, N_samples-2] Weights from coarse network (proportional to density)
        n_importance: Number of additional fine samples per ray (typically 128)
        perturb: If > 0, use stochastic sampling; if 0, deterministic
        pytest: If True, use fixed random seed for testing

    Returns:
        z_vals: [N_rays, N_samples + N_importance] Combined and sorted sample depths
        z_samples: [N_rays, N_importance] The new importance samples
    """
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_importance, det=(perturb < 1e-10), pytest=pytest)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    return z_vals, z_samples


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    n_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    n_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    pytest=False,
):
    """
    Volumetric rendering of a batch of rays using stratified and hierarchical sampling.

    This implements the core rendering algorithm described in Sections 4 and 5.2:

    1. Stratified Sampling (Section 4):
       - Divide ray into N_samples bins
       - Sample one point randomly within each bin
       - Query coarse network at each sample point

    2. Hierarchical Volume Sampling (Section 5.2):
       - Use coarse network weights to guide fine network sampling
       - Sample N_importance additional points in high-density regions
       - Query fine network with combined samples for final output

    This two-stage approach allows efficient sampling by focusing computation
    on relevant parts of the scene.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space (coarse network).
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of coarse samples along each ray (typically 64).
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth (disparity) rather than depth.
      perturb: float, 0 or 1. If non-zero, use stratified sampling with random jitter.
      N_importance: int. Number of additional fine samples along each ray (typically 128).
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: float. Standard deviation of noise added to sigma for regularization.
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # Extract ray information from the batch
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    # Create stratified samples along the ray
    z_vals = _create_stratified_samples(rays_o, near, far, n_samples, lindisp, perturb, pytest)

    # Compute 3D sample points along rays and query coarse network
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    # Render using volume rendering to get RGB, disparity, opacity, etc.
    rgb_map, disp_map, acc_map, weights, _depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    # ========== FINE NETWORK: Hierarchical Sampling ==========
    # If using hierarchical sampling (Section 5.2), use coarse weights to guide fine sampling
    if n_importance > 0:
        # Save coarse network outputs
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals, z_samples = _perform_hierarchical_sampling(z_vals, weights, n_importance, perturb, pytest)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        # Render with the fine network's predictions (these are the final outputs)
        rgb_map, disp_map, acc_map, weights, _depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if n_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument("--netdepth_fine", type=int, default=8, help="layers in fine network")
    parser.add_argument("--netwidth_fine", type=int, default=256, help="channels per layer in fine network")
    parser.add_argument(
        "--N_rand", type=int, default=32 * 32 * 4, help="batch size (number of random rays per gradient step)"
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential learning rate decay (in 1000 steps)")
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument("--no_batching", action="store_true", help="only take random rays from 1 image at a time")
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument(
        "--ft_path", type=str, default=None, help="specific weights npy file to reload for coarse network"
    )

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument("--N_importance", type=int, default=0, help="number of additional fine samples per ray")
    parser.add_argument("--perturb", type=float, default=1.0, help="set to 0. for no jitter, 1. for jitter")
    parser.add_argument("--use_viewdirs", action="store_true", help="use full 5D input instead of 3D")
    parser.add_argument("--i_embed", type=int, default=0, help="set 0 for default positional encoding, -1 for none")
    parser.add_argument(
        "--multires", type=int, default=10, help="log2 of max freq for positional encoding (3D location)"
    )
    parser.add_argument(
        "--multires_views", type=int, default=4, help="log2 of max freq for positional encoding (2D direction)"
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only", action="store_true", help="do not optimize, reload weights and render out render_poses path"
    )
    parser.add_argument("--render_test", action="store_true", help="render the test set instead of render_poses path")
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0, help="number of steps to train on central crops")
    parser.add_argument("--precrop_frac", type=float, default=0.5, help="fraction of img taken for central crops")

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="llff", help="options: llff / blender / deepvoxels")
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default="greek", help="options : armchair / cube / greek / vase")

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res", action="store_true", help="load blender synthetic data at 400x400 instead of 800x800"
    )

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, help="downsample factor for LLFF images")
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument("--lindisp", action="store_true", help="sampling linearly in disparity rather than depth")
    parser.add_argument("--spherify", action="store_true", help="set for spherical 360 scenes")
    parser.add_argument(
        "--llffhold", type=int, default=8, help="will take every 1/N images as LLFF test set, paper uses 8"
    )

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric loggin")
    parser.add_argument("--i_img", type=int, default=500, help="frequency of tensorboard image logging")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight ckpt saving")
    parser.add_argument("--i_testset", type=int, default=50000, help="frequency of testset saving")
    parser.add_argument("--i_video", type=int, default=50000, help="frequency of render_poses video saving")

    return parser


def _load_llff_dataset(args):
    """Load LLFF dataset."""
    images, poses, bds, render_poses, i_test = load_llff_data(
        args.datadir, args.factor, recenter=True, bd_factor=0.75, spherify=args.spherify
    )
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)

    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print("Auto LLFF holdout,", args.llffhold)
        i_test = np.arange(images.shape[0])[:: args.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

    print("DEFINING BOUNDS")
    if args.no_ndc:
        near = np.ndarray.min(bds) * 0.9
        far = np.ndarray.max(bds) * 1.0
    else:
        near = 0.0
        far = 1.0
    print("NEAR FAR", near, far)

    return images, poses, hwf, render_poses, i_train, i_val, i_test, None, near, far


def _load_blender_dataset(args):
    """Load Blender dataset."""
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    near = 2.0
    far = 6.0
    images = (images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])) if args.white_bkgd else images[..., :3]
    return images, poses, hwf, render_poses, i_train, i_val, i_test, None, near, far


def _load_linemod_dataset(args):
    """Load LINEMOD dataset."""
    images, poses, render_poses, hwf, K, i_split, near, far = load_linemod_dataset(
        args.datadir, args.half_res, args.testskip
    )
    print(f"Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}")
    print(f"[CHECK HERE] near: {near}, far: {far}.")
    i_train, i_val, i_test = i_split
    images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:]) if args.white_bkgd else images[..., :3]
    return images, poses, hwf, render_poses, i_train, i_val, i_test, K, near, far


def _load_deepvoxels_dataset(args):
    """Load DeepVoxels dataset."""
    images, poses, render_poses, hwf, i_split = load_dv_data(
        scene=args.shape, basedir=args.datadir, testskip=args.testskip
    )
    print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    hemi_r = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
    near = hemi_r - 1.0
    far = hemi_r + 1.0
    return images, poses, hwf, render_poses, i_train, i_val, i_test, None, near, far


def load_dataset(args):
    """
    Load dataset based on dataset type and return relevant data.

    Supports four dataset types:
    1. LLFF (Local Light Field Fusion): Real-world forward-facing scenes
       - 20-30 images per scene
       - Uses NDC ray parameterization
       - Camera poses estimated from COLMAP

    2. Blender: Synthetic scenes with perfect ground truth
       - Rendered images with known camera poses
       - Clean backgrounds, controlled lighting
       - Used for quantitative evaluation

    3. LINEMOD: Object recognition dataset
       - Object-centric captures
       - Intrinsic camera matrix provided

    4. DeepVoxels: Synthetic object-centric data
       - Hemispheric camera arrangement
       - Voxel-based baseline comparisons

    Args:
        args: Parsed arguments containing dataset_type and data paths

    Returns:
        images: [N, H, W, 3] RGB images
        poses: [N, 3, 4] Camera-to-world transformation matrices
        hwf: [3] Height, width, focal length
        render_poses: Poses for novel view rendering
        i_train: Indices of training images
        i_val: Indices of validation images
        i_test: Indices of test images
        K: Camera intrinsic matrix (or None)
        near: Near plane distance
        far: Far plane distance
    """
    dataset_loaders = {
        "llff": _load_llff_dataset,
        "blender": _load_blender_dataset,
        "LINEMOD": _load_linemod_dataset,
        "deepvoxels": _load_deepvoxels_dataset,
    }

    loader = dataset_loaders.get(args.dataset_type)
    if loader is None:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return None

    return loader(args)


def setup_logging_dirs(basedir, expname, args):
    """
    Create log directories and save config files for experiment tracking.

    This ensures reproducibility by:
    1. Creating experiment directory structure
    2. Saving all command-line arguments
    3. Saving config file for future reference

    Directory structure created:
    logs/
    └── {expname}/
        ├── args.txt          # All hyperparameters
        ├── config.txt        # Original config file
        ├── {iter:06d}.tar   # Model checkpoints
        └── testset_{iter}/  # Rendered test images

    Args:
        basedir: Base directory for logs (typically ./logs/)
        expname: Experiment name (used as subdirectory)
        args: Parsed arguments to save
    """
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # Save args
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f"{arg} = {attr}\n")

    # Save config file if provided
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file, open(args.config) as config_file:
            file.write(config_file.read())


def handle_render_only_mode(
    args, render_poses, hwf, focus, render_kwargs_test, basedir, expname, start, images, i_test
):
    """
    Handle render-only mode execution (no training).

    This mode is used to:
    1. Generate novel view synthesis videos from trained models
    2. Render test set images for quantitative evaluation
    3. Create visualizations of learned scene representations

    Invoked with: python run_nerf.py --config configs/scene.txt --render_only

    The function loads a trained checkpoint and renders either:
    - Test set images (if --render_test flag is set)
    - Novel view path (spiral/sphere trajectory otherwise)

    Args:
        args: Parsed arguments
        render_poses: Camera poses for rendering
        hwf: Height, width, focal length
        focus: Camera intrinsic matrix
        render_kwargs_test: Rendering parameters for test mode
        basedir: Base directory for logs
        expname: Experiment name
        start: Starting iteration (from loaded checkpoint)
        images: Dataset images
        i_test: Test set indices
    """
    print("RENDER ONLY")
    with torch.no_grad():
        images = images[i_test] if args.render_test else None
        testsavedir = os.path.join(
            basedir, expname, "renderonly_{}_{:06d}".format("test" if args.render_test else "path", start)
        )
        os.makedirs(testsavedir, exist_ok=True)
        print("test poses shape", render_poses.shape)

        rgbs, _ = render_path(
            render_poses,
            hwf,
            focus,
            args.chunk,
            render_kwargs_test,
            savedir=testsavedir,
            render_factor=args.render_factor,
        )
        print("Done rendering", testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8)


def prepare_ray_batching(use_batching, image_height, image_width, focus, poses, images, i_train):
    """
    Prepare ray batching data structures for efficient training.

    When ray batching is enabled, this function:
    1. Pre-computes rays for all pixels in all training images
    2. Concatenates rays with their corresponding RGB values
    3. Shuffles the rays to ensure random sampling across images
    4. Creates a large pool of [ray_origin, ray_direction, RGB] tuples

    This allows sampling random rays from across the entire training set,
    which can improve convergence but requires significant GPU memory.

    Args:
        use_batching: Whether to use ray batching (False if --no_batching flag set)
        image_height: Image height in pixels
        image_width: Image width in pixels
        focus: Camera intrinsic matrix
        poses: [N, 3, 4] Camera poses
        images: [N, H, W, 3] Training images
        i_train: Indices of training images

    Returns:
        rays_rgb: [N_rays, 3, 3] Pre-computed rays and RGB values, or None
        i_batch: Starting batch index (0)
    """
    if not use_batching:
        return None, 0

    print("get rays")
    rays = np.stack([get_rays_np(image_height, image_width, focus, p) for p in poses[:, :3, :4]], 0)
    print("done, concats")
    rays_rgb = np.concatenate([rays, images[:, None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
    rays_rgb = rays_rgb.astype(np.float32)
    print("shuffle rays")
    np.random.shuffle(rays_rgb)
    print("done")
    return rays_rgb, 0


def sample_ray_batch(
    use_batching, rays_rgb, i_batch, n_rand, images, poses, i_train, image_height, image_width, focus, args, i, start
):
    """
    Sample a batch of rays for training.

    Two sampling strategies are supported:
    1. Ray batching (use_batching=True):
       - Pre-compute all rays from all training images
       - Sample random rays from the entire training set
       - More memory intensive but ensures diverse ray samples

    2. Image-based sampling (use_batching=False, no_batching flag):
       - Select one random training image
       - Sample random rays only from that image
       - Less memory but may slow convergence

    The paper uses image-based sampling by default. Ray batching can improve
    convergence for some scenes but requires more GPU memory.

    Args:
        use_batching: Whether to use pre-computed ray batches
        rays_rgb: Pre-computed rays and RGB values (if batching)
        i_batch: Current batch index (if batching)
        n_rand: Number of random rays to sample (typically 4096)
        images: Training images
        poses: Camera poses
        i_train: Indices of training images
        image_height: Image height
        image_width: Image width
        focus: Camera intrinsic matrix
        args: Training arguments (contains precrop settings)
        i: Current training iteration
        start: Starting iteration (for precrop)

    Returns:
        batch_rays: [2, N_rand, 3] Ray origins and directions
        target_s: [N_rand, 3] Ground truth RGB for sampled rays
        rays_rgb: Updated pre-computed rays (if batching)
        i_batch: Updated batch index (if batching)
    """
    if use_batching:
        batch = rays_rgb[i_batch : i_batch + n_rand]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += n_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        return batch_rays, target_s, rays_rgb, i_batch
    else:
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        if n_rand is not None:
            rays_o, rays_d = get_rays(image_height, image_width, focus, torch.tensor(pose, device=device))
            coords = _get_sampling_coords(image_height, image_width, i, args, start)
            select_inds = np.random.choice(coords.shape[0], size=[n_rand], replace=False)
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]

        return batch_rays, target_s, rays_rgb, i_batch


def _get_sampling_coords(image_height, image_width, i, args, start):
    """
    Get coordinate sampling grid for ray selection with optional center cropping.

    During early training iterations, center cropping can help:
    1. Focus learning on the central object/content first
    2. Avoid wasting computation on background in early iterations
    3. Stabilize training by learning easier central regions first

    After precrop_iters iterations, sampling expands to the full image.

    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        i: Current training iteration
        args: Arguments containing precrop_iters and precrop_frac
        start: Starting iteration (for logging)

    Returns:
        coords: [N_pixels, 2] Flattened pixel coordinates available for sampling
    """
    if i < args.precrop_iters:
        crop_height = int(image_height // 2 * args.precrop_frac)
        crop_width = int(image_width // 2 * args.precrop_frac)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    image_height // 2 - crop_height, image_height // 2 + crop_height - 1, 2 * crop_height, device=device
                ),
                torch.linspace(
                    image_width // 2 - crop_width, image_width // 2 + crop_width - 1, 2 * crop_width, device=device
                ),
            ),
            -1,
        )
        if i == start:
            print(
                f"[Config] Center cropping of size {2 * crop_height} x {2 * crop_width} is enabled until iter {args.precrop_iters}"
            )
    else:
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, image_height - 1, image_height, device=device),
                torch.linspace(0, image_width - 1, image_width, device=device),
            ),
            -1,
        )

    return torch.reshape(coords, [-1, 2])


def compute_training_loss(rgb, target_s, extras):
    """
    Compute photometric training loss from rendered RGB and ground truth.

    The loss function is a simple mean squared error (MSE) between rendered pixels
    and ground truth pixels. When using hierarchical sampling, the loss includes
    contributions from both the coarse and fine networks:

    L = MSE(C_fine, C_gt) + MSE(C_coarse, C_gt)

    This dual loss helps train both networks and ensures the coarse network provides
    useful guidance for hierarchical sampling.

    Args:
        rgb: [N_rays, 3] Rendered RGB from fine network
        target_s: [N_rays, 3] Ground truth RGB values
        extras: Dictionary potentially containing 'rgb0' (coarse network output)

    Returns:
        loss: Combined MSE loss (fine + coarse if hierarchical)
        psnr: Peak Signal-to-Noise Ratio in dB (quality metric)
    """
    img_loss = img2mse(rgb, target_s)
    loss = img_loss
    psnr = mse2psnr(img_loss)

    if "rgb0" in extras:
        img_loss0 = img2mse(extras["rgb0"], target_s)
        loss = loss + img_loss0

    return loss, psnr


def update_learning_rate(optimizer, args, global_step):
    """
    Update learning rate with exponential decay.

    The learning rate follows an exponential decay schedule:
    lr(t) = lr_init × (0.1)^(t / decay_steps)

    where decay_steps = lrate_decay × 1000 (typically 250,000 steps).

    This gradual learning rate reduction helps:
    1. Make large updates early in training for rapid convergence
    2. Make fine-grained updates later for detail refinement
    3. Stabilize training as the model approaches convergence

    Args:
        optimizer: PyTorch Adam optimizer
        args: Arguments containing lrate (initial rate) and lrate_decay
        global_step: Current training iteration
    """
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate


def save_checkpoint(i, basedir, expname, global_step, render_kwargs_train, optimizer):
    """
    Save model checkpoint for resuming training or inference.

    Checkpoints contain:
    - Network weights (coarse and fine)
    - Optimizer state (for seamless training resumption)
    - Global step counter

    This allows:
    1. Resuming training after interruption
    2. Loading trained models for novel view synthesis
    3. Fine-tuning from pre-trained checkpoints

    Args:
        i: Current iteration
        basedir: Base directory for logs
        expname: Experiment name
        global_step: Global training step counter
        render_kwargs_train: Dictionary containing network models
        optimizer: Adam optimizer with state
    """
    path = os.path.join(basedir, expname, f"{i:06d}.tar")
    torch.save(
        {
            "global_step": global_step,
            "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
            "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print("Saved checkpoints at", path)


def save_video_outputs(i, basedir, expname, render_poses, hwf, focus, args, render_kwargs_test):
    """Save video outputs during training."""
    with torch.no_grad():
        rgbs, disps = render_path(render_poses, hwf, focus, args.chunk, render_kwargs_test)
    print("Done, saving", rgbs.shape, disps.shape)
    moviebase = os.path.join(basedir, expname, f"{expname}_spiral_{i:06d}_")
    imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8)


def save_test_outputs(i, basedir, expname, poses, i_test, hwf, focus, args, render_kwargs_test):
    """Save test set outputs during training."""
    testsavedir = os.path.join(basedir, expname, f"testset_{i:06d}")
    os.makedirs(testsavedir, exist_ok=True)
    print("test poses shape", poses[i_test].shape)
    with torch.no_grad():
        render_path(
            torch.Tensor(poses[i_test]).to(device),
            hwf,
            focus,
            args.chunk,
            render_kwargs_test,
            savedir=testsavedir,
        )
    print("Saved test set")


def _handle_periodic_logging(
    i,
    args,
    training_state,
    scene_data,
    loss,
    psnr,
):
    """Handle periodic saves and logging during training."""
    if i % args.i_weights == 0:
        save_checkpoint(
            i,
            training_state["basedir"],
            training_state["expname"],
            training_state["global_step"],
            training_state["render_kwargs_train"],
            training_state["optimizer"],
        )

    if i % args.i_video == 0 and i > 0:
        save_video_outputs(
            i,
            training_state["basedir"],
            training_state["expname"],
            scene_data["render_poses"],
            scene_data["hwf"],
            scene_data["focus"],
            args,
            training_state["render_kwargs_test"],
        )

    if i % args.i_testset == 0 and i > 0:
        save_test_outputs(
            i,
            training_state["basedir"],
            training_state["expname"],
            scene_data["poses"],
            scene_data["i_test"],
            scene_data["hwf"],
            scene_data["focus"],
            args,
            training_state["render_kwargs_test"],
        )

    if i % args.i_print == 0:
        tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")


def _prepare_training_data(use_batching, images, poses, rays_rgb):
    """Move training data to GPU."""
    if use_batching:
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
    poses = torch.Tensor(poses).to(device)
    return images, poses, rays_rgb


def train():
    """
    Main training function for Neural Radiance Fields (NeRF).

    This function implements the complete NeRF training pipeline as described in the paper:
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"

    Training Pipeline:
    1. Load dataset (synthetic Blender, real LLFF, DeepVoxels, or LINEMOD)
    2. Initialize coarse and fine networks with positional encoding
    3. For 200k iterations:
        a. Sample random rays from training images
        b. Stratified sampling: Sample N_samples (64) points along each ray
        c. Query coarse network at sample points
        d. Hierarchical sampling: Sample N_importance (128) additional points using coarse weights
        e. Query fine network at all points
        f. Volume rendering: Compute RGB using classical volume rendering (Section 4)
        g. Compute photometric loss: MSE between rendered and ground truth RGB
        h. Backpropagate and update weights with Adam optimizer
        i. Periodically save checkpoints, render test views, and generate videos

    Key Components:
    - Positional Encoding (Section 5.1): Maps 3D coordinates to higher dimensions
    - Volume Rendering (Section 4): Accumulates color and density along rays
    - Hierarchical Sampling (Section 5.2): Two-stage coarse-to-fine sampling
    - View-Dependent Effects: Conditions color on viewing direction for specularities

    The result is a continuous 5D function (x, y, z, θ, φ) → (R, G, B, σ) that can
    synthesize photorealistic novel views of the scene.

    Returns:
        None (saves checkpoints and rendered outputs to disk)
    """
    parser = config_parser()
    args = parser.parse_args()

    # Load dataset
    dataset_data = load_dataset(args)
    if dataset_data is None:
        return

    images, poses, hwf, render_poses, i_train, i_val, i_test, K, near, far = dataset_data

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Setup logging directories
    basedir = args.basedir
    expname = args.expname
    setup_logging_dirs(basedir, expname, args)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, _grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {"near": near, "far": far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Handle render-only mode
    if args.render_only:
        handle_render_only_mode(args, render_poses, hwf, K, render_kwargs_test, basedir, expname, start, images, i_test)
        return

    # Prepare ray batching
    n_rand = args.N_rand
    use_batching = not args.no_batching
    rays_rgb, i_batch = prepare_ray_batching(use_batching, H, W, K, poses, images, i_train)

    # Move training data to GPU
    images, poses, rays_rgb = _prepare_training_data(use_batching, images, poses, rays_rgb)

    # Training loop setup
    n_iters = 200000 + 1
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    start = start + 1
    for i in trange(start, n_iters):
        # Sample random ray batch
        batch_rays, target_s, rays_rgb, i_batch = sample_ray_batch(
            use_batching, rays_rgb, i_batch, n_rand, images, poses, i_train, H, W, K, args, i, start
        )

        # Render and compute loss
        rgb, _disp, _acc, extras = render(
            H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train
        )

        optimizer.zero_grad()
        loss, psnr = compute_training_loss(rgb, target_s, extras)
        loss.backward()
        optimizer.step()

        # Update learning rate
        update_learning_rate(optimizer, args, global_step)

        # Periodic saves and logging
        training_state = {
            "basedir": basedir,
            "expname": expname,
            "global_step": global_step,
            "render_kwargs_train": render_kwargs_train,
            "render_kwargs_test": render_kwargs_test,
            "optimizer": optimizer,
        }
        scene_data = {
            "render_poses": render_poses,
            "hwf": hwf,
            "focus": K,
            "poses": poses,
            "i_test": i_test,
        }
        _handle_periodic_logging(
            i,
            args,
            training_state,
            scene_data,
            loss,
            psnr,
        )

        global_step += 1


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")

    train()
