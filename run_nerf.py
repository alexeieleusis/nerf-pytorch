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
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
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
    """Render rays in smaller minibatches to avoid OOM."""
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
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
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
    """Instantiate NeRF's MLP model."""
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
    """Create stratified samples along rays."""
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
    """Perform hierarchical sampling using coarse network weights."""
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
    """Load dataset based on dataset type and return relevant data."""
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
    """Create log directories and save config files."""
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
    """Handle render-only mode execution."""
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
    """Prepare ray batching data structures."""
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
    """Sample a batch of rays for training."""
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
    """Get coordinate sampling grid for ray selection."""
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
    """Compute training loss from rendered RGB and ground truth."""
    img_loss = img2mse(rgb, target_s)
    loss = img_loss
    psnr = mse2psnr(img_loss)

    if "rgb0" in extras:
        img_loss0 = img2mse(extras["rgb0"], target_s)
        loss = loss + img_loss0

    return loss, psnr


def update_learning_rate(optimizer, args, global_step):
    """Update learning rate with exponential decay."""
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate


def save_checkpoint(i, basedir, expname, global_step, render_kwargs_train, optimizer):
    """Save model checkpoint."""
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
    basedir,
    expname,
    global_step,
    render_kwargs_train,
    render_kwargs_test,
    optimizer,
    render_poses,
    hwf,
    focus,
    poses,
    i_test,
    loss,
    psnr,
):
    """Handle periodic saves and logging during training."""
    if i % args.i_weights == 0:
        save_checkpoint(i, basedir, expname, global_step, render_kwargs_train, optimizer)

    if i % args.i_video == 0 and i > 0:
        save_video_outputs(i, basedir, expname, render_poses, hwf, focus, args, render_kwargs_test)

    if i % args.i_testset == 0 and i > 0:
        save_test_outputs(i, basedir, expname, poses, i_test, hwf, focus, args, render_kwargs_test)

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
        _handle_periodic_logging(
            i,
            args,
            basedir,
            expname,
            global_step,
            render_kwargs_train,
            render_kwargs_test,
            optimizer,
            render_poses,
            hwf,
            K,
            poses,
            i_test,
            loss,
            psnr,
        )

        global_step += 1


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")

    train()
