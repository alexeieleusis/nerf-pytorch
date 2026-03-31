"""
NeRF Helper Functions and Neural Network Architecture
=====================================================

This module contains the core building blocks for Neural Radiance Fields (NeRF):

1. Positional Encoding (Section 5.1):
   - Embedder class: Implements gamma(p) function
   - Maps low-dimensional coordinates to high-dimensional space
   - Enables learning of high-frequency variations in geometry and color

2. NeRF MLP Architecture (Section 3):
   - 8-layer fully-connected network with skip connections
   - Separate branches for density (view-independent) and color (view-dependent)
   - Processes encoded 5D input: position (x,y,z) + viewing direction (θ,φ)

3. Ray Generation and Sampling:
   - Camera ray generation from pinhole camera model
   - NDC coordinate transformation for forward-facing scenes
   - Hierarchical sampling using inverse transform sampling (Section 5.2)

4. Utility Functions:
   - img2mse: Mean squared error loss
   - mse2psnr: Convert MSE to Peak Signal-to-Noise Ratio
   - to8b: Convert float images to 8-bit for saving
"""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F

# Misc utility functions
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (Section 5.1 of the paper)
# This implements the γ(p) function from equation (4), which maps continuous input coordinates
# to a higher dimensional space using high frequency functions. This enables the MLP to
# represent high-frequency variations in color and geometry.
#
# The encoding is: γ(p) = (sin(2^0·π·p), cos(2^0·π·p), sin(2^1·π·p), cos(2^1·π·p), ..., sin(2^(L-1)·π·p), cos(2^(L-1)·π·p))
# where L is the number of frequency bands (multires hyperparameter).
# According to the paper: L=10 for spatial position x, and L=4 for viewing direction d.
class Embedder:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.embed_fns: List[Callable[[torch.Tensor], torch.Tensor]] = []
        self.out_dim: int = 0
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns: List[Callable[[torch.Tensor], torch.Tensor]] = []
        d: int = self.kwargs[
            "input_dims"
        ]  # Always 3: for position (x,y,z) or viewing direction as 3D Cartesian unit vector d
        out_dim = 0

        # Option to include the original input along with the encoded version
        if self.kwargs["include_input"]:
            identity_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
            embed_fns.append(identity_fn)
            out_dim += d

        max_freq: int = self.kwargs["max_freq_log2"]  # L-1, where L is number of frequency bands
        n_freqs: int = self.kwargs["num_freqs"]  # L, number of frequency bands

        # Create frequency bands: 2^0, 2^1, 2^2, ..., 2^(L-1)
        # Log sampling means we sample frequencies logarithmically
        # Note: freq_bands are created on CPU, but will be moved to correct device during embed
        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=n_freqs)

        # For each frequency band, apply both sin and cos
        # FIXME: Ask Claude to explain how each variable in the formula maps to code, and if freq=freq_val: p_fn(2.0 * np.pi * freq * x) should be freq=freq_val: p_fn(2.0 ** (some_var) * np.pi * freq * x), in the loop below.
        # This creates: [sin(2^0*π*x), cos(2^0*π*x), sin(2^1*π*x), cos(2^1*π*x), ...]
        # Formula from paper equation (4): γ(p) = (sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
        # We convert freq to float to avoid device issues (scalars work on any device)
        for freq in freq_bands:
            freq_val = freq.item()
            for p_fn in self.kwargs["periodic_fns"]:  # [sin, cos]
                periodic_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x, p_fn=p_fn, freq=freq_val: p_fn(
                    2.0 * np.pi * freq * x
                )
                embed_fns.append(periodic_fn)
                out_dim += d  # Each periodic function adds d dimensions

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        # Apply all embedding functions and concatenate results
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires: int, i: int = 0) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """
    Factory function to create a positional encoding embedder.

    Args:
        multires: Number of frequency bands L (typically 10 for position, 4 for viewing direction)
        i: Embedding type. -1 for no encoding (identity), 0 for default positional encoding

    Returns:
        embed: Function that takes coordinates and returns positionally encoded values
        out_dim: Output dimensionality of the encoding
    """
    if i == -1:
        # No positional encoding, just pass through the input
        return nn.Identity(), 3

    # Configuration for positional encoding
    # With include_input=True, output is: [x, sin(2^0πx), cos(2^0πx), ..., sin(2^(L-1)πx), cos(2^(L-1)πx)]
    # Output dimension = 3 + 3*2*L = 3 + 6L (for 3D input)
    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,  # L-1
        "num_freqs": multires,  # L
        "log_sampling": True,  # Use logarithmic frequency sampling
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed: Callable[[torch.Tensor], torch.Tensor] = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) MLP architecture.

    This implements the network F_Θ described in Section 3 of the paper.
    The network takes as input a 5D coordinate (spatial position x=(x,y,z) and viewing
    direction d as a 3D Cartesian unit vector) and outputs volume density σ and view-dependent
    RGB color c = (r,g,b).

    Architecture details from paper (Section 3, Appendix A - Figure 7):
    - 8 fully-connected layers (D=8) with ReLU activations, 256 channels per layer (W=256)
    - Skip connection at layer 5 (index 4): concatenates positionally encoded input with layer 4 activations
    - Positional encoding γ(·) applied separately to position x (with L=10) and direction d (with L=4)
    - Volume density σ depends only on position x (view-independent, ensures multiview consistency)
    - A 256-D feature vector is concatenated with encoded viewing direction γ(d)
    - One additional 128-channel fully-connected ReLU layer processes the combined features
    - Final layer outputs view-dependent RGB color c (enables modeling of specular reflections)

    The network computes: F_Θ(γ(x), γ(d)) = (c, σ)
    where the representation is multiview-consistent by restricting σ = f(γ(x)) only.
    """

    def __init__(
        self,
        depth: int = 8,
        width: int = 256,
        input_ch: int = 3,
        input_ch_views: int = 3,
        output_ch: int = 4,
        skips: Optional[List[int]] = None,
        use_viewdirs: bool = False,
    ) -> None:
        """
        Args:
            depth: Number of layers in the main MLP (default: 8, denoted as D in paper)
            width: Width (number of channels) of each layer (default: 256, denoted as W in paper)
            input_ch: Number of input channels for position (63 with positional encoding, L=10)
            input_ch_views: Number of input channels for viewing direction (27 with encoding, L=4)
            output_ch: Number of output channels (4 for RGB+density, or 5 for coarse/fine models)
            skips: List of layer indices to add skip connections (typically [4] for skip at layer 5)
            use_viewdirs: Whether to use viewing direction as input (enables view-dependent effects)
        """
        super().__init__()
        self.D = depth
        self.W = width
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips if skips is not None else [4]
        self.use_viewdirs = use_viewdirs

        # Main MLP for processing position
        # Consists of D layers with skip connections at specified layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, width)]
            + [
                nn.Linear(width, width) if i not in self.skips else nn.Linear(width + input_ch, width)
                for i in range(depth - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # Additional MLP for processing viewing direction (single layer in official implementation)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + width, width // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            # When using viewing directions, split the network:
            # - alpha (density sigma) depends only on position
            # - rgb (color c) depends on position and viewing direction
            self.feature_linear = nn.Linear(width, width)
            self.alpha_linear = nn.Linear(width, 1)  # Outputs volume density sigma
            self.rgb_linear = nn.Linear(width // 2, 3)  # Outputs RGB color c
        else:
            # Simple case: directly output RGB+density from position
            self.output_linear = nn.Linear(width, output_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NeRF network.

        The input x contains concatenated positionally encoded coordinates:
        x = [γ(x), γ(d)] where γ is the positional encoding function from Section 5.1.

        Args:
            x: [batch, input_ch + input_ch_views] Concatenated encoded position and viewing direction

        Returns:
            outputs: [batch, 4] tensor containing [R, G, B, σ] where:
                     R, G, B ∈ [0,1] are color channels (after sigmoid activation)
                     σ ≥ 0 is the volume density (after ReLU activation)
        """
        # Split input into position and viewing direction components
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # Process through main MLP layers with skip connections
        # Skip connections help the network learn high-frequency details
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # Concatenate original input at skip layer (typically layer 5)
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Separate path for density (view-independent) and color (view-dependent)
            # This is key to modeling view-dependent effects like specularities
            alpha = self.alpha_linear(h)  # Volume density sigma (view-independent)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)  # Concatenate viewing direction

            # Process through view-dependent layers
            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)  # RGB color c (view-dependent)
            outputs = torch.cat([rgb, alpha], -1)  # [R, G, B, sigma]
        else:
            # Simple case: both RGB and density from position only
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights: List[npt.NDArray[np.floating[Any]]]) -> None:
        if not self.use_viewdirs:
            raise NotImplementedError("load_weights_from_keras is not implemented if use_viewdirs=False")

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(
    image_height: int, image_width: int, focal: torch.Tensor, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ray origins and directions for all pixels in an image using pinhole camera model.

    This function implements the perspective camera ray generation. Each pixel corresponds to
    a ray in 3D space defined parametrically as: r(t) = o + t·d, where:
    - o is the ray origin (camera center in world coordinates)
    - d is the ray direction (unit vector in world coordinates)
    - t ≥ 0 is the distance along the ray from the origin

    Args:
        image_height: Image height in pixels (H in paper notation)
        image_width: Image width in pixels (W in paper notation)
        focal: Camera intrinsic matrix K [3x3] containing:
               K = [[fx,  0, cx],
                    [ 0, fy, cy],
                    [ 0,  0,  1]]
               where (fx, fy) are focal lengths and (cx, cy) is the principal point
        c2w: Camera-to-world transformation matrix [3x4] (extrinsic parameters)
             Transforms points from camera space to world space

    Returns:
        rays_o: [H, W, 3] Ray origins (all equal to camera center in world coordinates)
        rays_d: [H, W, 3] Ray directions in world coordinates (not normalized)
    """
    # Create pixel coordinate grid
    # Determine device from c2w matrix
    device = c2w.device if isinstance(c2w, torch.Tensor) else torch.device("cpu")
    i, j = torch.meshgrid(
        torch.linspace(0, image_width - 1, image_width, device=device),
        torch.linspace(0, image_height - 1, image_height, device=device),
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()  # Transpose to get correct [H, W] shape
    j = j.t()

    # Convert pixel coordinates to normalized camera coordinates using intrinsics
    # K[0][0] = focal_x, K[1][1] = focal_y, K[0][2] = cx, K[1][2] = cy
    # This gives us ray directions in the camera coordinate system
    dirs = torch.stack([(i - focal[0][2]) / focal[0][0], -(j - focal[1][2]) / focal[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(
    image_height: int, image_width: int, focal: npt.NDArray[np.floating[Any]], c2w: npt.NDArray[np.floating[Any]]
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Generate ray origins and directions for all pixels in an image (NumPy version).

    This is the NumPy equivalent of get_rays(), used for preprocessing and batching
    during data loading. Implements the same pinhole camera model.

    Args:
        image_height: Image height in pixels (H)
        image_width: Image width in pixels (W)
        focal: Camera intrinsic matrix K [3x3] containing focal lengths and principal point
        c2w: Camera-to-world transformation matrix [3x4] (extrinsic parameters)

    Returns:
        rays_o: [H, W, 3] Ray origins in world coordinates (NumPy array)
        rays_d: [H, W, 3] Ray directions in world coordinates (NumPy array, not normalized)
    """
    i, j = np.meshgrid(
        np.arange(image_width, dtype=np.float32), np.arange(image_height, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack([(i - focal[0][2]) / focal[0][0], -(j - focal[1][2]) / focal[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(
    image_height: int, image_width: int, focal: float, near: float, rays_o: torch.Tensor, rays_d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform rays from world coordinates to Normalized Device Coordinates (NDC).

    This transformation is described in Appendix C of the paper and is used specifically
    for forward-facing scenes (like those in the LLFF dataset) to better handle unbounded
    scenes. NDC space normalizes the viewing frustum into a canonical coordinate system
    where:
    - The camera looks down the -Z axis
    - The viewing frustum is mapped to a unit cube [-1,1]³
    - Depth is parameterized by disparity (inverse depth) rather than linear depth
    - The near plane maps to depth = 1, and infinity maps to depth = -1

    The transformation consists of three steps:
    1. Shift ray origins to the near plane: o' = o + t_near·d where t_near = -(near + o_z)/d_z
    2. Apply perspective projection to X and Y coordinates
    3. Reparameterize depth to use disparity (1/depth) for better sampling distribution

    Benefits for forward-facing captures:
    - Handles unbounded backgrounds (depth → ∞) naturally
    - More uniform sampling distribution across the depth range
    - Better numerical stability for scenes with large depth variation
    - Matches the coordinate system used by Local Light Field Fusion (LLFF)

    Reference: NeRF paper Appendix C and LLFF paper [Mildenhall et al. 2019].

    Args:
        image_height: Image height H in pixels
        image_width: Image width W in pixels
        focal: Focal length f of the camera
        near: Near plane distance n
        rays_o: [N_rays, 3] Ray origins in world coordinates
        rays_d: [N_rays, 3] Ray directions in world coordinates

    Returns:
        rays_o: [N_rays, 3] Transformed ray origins in NDC space
        rays_d: [N_rays, 3] Transformed ray directions in NDC space
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (image_width / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (image_height / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (image_width / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (image_height / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(
    bins: torch.Tensor, weights: torch.Tensor, n_samples: int, det: bool = False, pytest: bool = False
) -> torch.Tensor:
    """
    Hierarchical sampling using inverse transform sampling.

    This implements the hierarchical volume sampling strategy described in Section 5.2.
    The key idea: sample more points in regions where we expect more content (higher weight).

    The algorithm:
    1. Convert weights to a probability distribution (PDF)
    2. Compute cumulative distribution function (CDF)
    3. Sample uniformly from [0, 1] and invert the CDF to get sample locations
    4. This concentrates samples in high-weight regions

    This is used by the "fine" network to focus on relevant parts of the volume
    based on the coarse network's density predictions.

    Args:
        bins: [N_rays, N_samples-1] Bin centers from coarse sampling
        weights: [N_rays, N_samples-2] Weights from coarse network (proportional to density)
        N_samples: Number of new samples to draw
        det: If True, use deterministic sampling; if False, use random sampling
        pytest: If True, use fixed random seed for reproducibility

    Returns:
        samples: [N_rays, N_samples] New sample locations along each ray
    """
    # --- Stage 1: Build a probability density function (PDF) from the coarse weights ---
    # The coarse network outputs a weight per bin that is proportional to how much
    # volume density (and thus expected color contribution) is concentrated there.
    # We add a small epsilon to avoid zero weights, which would cause division by zero
    # and produce NaN values in the PDF.
    weights = weights + 1e-5  # prevent nans and ensure all weights are positive
    # Divide each weight by the total so the values sum to 1 across bins, making it a valid PDF.
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # --- Stage 2: Build the Cumulative Distribution Function (CDF) ---
    # The CDF at position i is the sum of all PDF values up to and including i.
    # It is a monotonically increasing function from 0 to 1.
    # Example: if pdf = [0.1, 0.6, 0.3], then cdf = [0.1, 0.7, 1.0]
    cdf = torch.cumsum(pdf, -1)
    # Prepend a zero so the CDF starts at 0, making it span exactly [0, 1].
    # This ensures every uniform sample u ∈ [0, 1] can be mapped to a bin.
    # After cat: cdf = [0.0, 0.1, 0.7, 1.0] for the example above.
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # --- Stage 3: Draw uniform random samples in [0, 1] ---
    # These are the "query points" we will invert through the CDF.
    # Each sample u represents a quantile: we want to find the bin depth t such that
    # CDF(t) = u, which by construction samples bins proportionally to their weight.
    u: torch.Tensor
    if det:
        # Deterministic mode: evenly spaced quantiles across [0, 1].
        # Used during evaluation/rendering for consistent, repeatable results.
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=cdf.device)
        u = u.expand([*list(cdf.shape[:-1]), n_samples])
    else:
        # Stochastic mode: independent uniform random samples per ray.
        # Used during training to introduce randomness and avoid aliasing.
        u = torch.rand([*list(cdf.shape[:-1]), n_samples], device=cdf.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = [*list(cdf.shape[:-1]), n_samples]
        if det:
            u_np: npt.NDArray[np.floating[Any]] = np.linspace(0.0, 1.0, n_samples)
            u_np = np.broadcast_to(u_np, new_shape)
        else:
            u_np = np.random.rand(*new_shape)
        u = torch.tensor(u_np, device=cdf.device)

    # --- Stage 4: Invert the CDF via binary search (inverse transform sampling) ---
    # For each uniform sample u, find the index in the CDF where u would be inserted
    # to keep it sorted. This gives us the bin that the quantile u falls into.
    # right=True means we use the right boundary when u equals a CDF value exactly.
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    # Clamp the indices so they stay within valid bounds: [0, len(cdf)-1].
    # 'below' is the index of the left edge of the bracket containing u.
    # 'above' is the index of the right edge of the bracket containing u.
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # Stack into pairs (below, above) so we can gather both edges at once.
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # --- Stage 5: Gather the CDF and bin values at the bracket edges ---
    # We need the CDF values and bin depths at both the left and right edges of
    # the bracket so we can interpolate between them.
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # The expand+gather pattern is the PyTorch equivalent of TensorFlow's batched gather.
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # --- Stage 6: Linear interpolation to find the exact sample depth ---
    # We know u lies somewhere between cdf_g[...,0] and cdf_g[...,1].
    # We find the fractional position t of u within that CDF interval,
    # then apply the same fraction to the corresponding bin depth interval.
    # Guard against degenerate brackets (zero-width CDF interval) by replacing
    # the denominator with 1, which makes t=0 and returns the left bin edge.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom  # fractional position within the bracket [0, 1]
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # interpolated depth

    return samples
