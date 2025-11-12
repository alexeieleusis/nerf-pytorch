import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
# This implements the γ(p) function from the paper, which maps continuous input coordinates
# to a higher dimensional space using high frequency functions. This helps the network learn
# high-frequency variations in color and geometry.
#
# The encoding is: γ(p) = (sin(2^0πp), cos(2^0πp), sin(2^1πp), cos(2^1πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
# where L is the number of frequency bands (multires hyperparameter)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # 3 for position (x,y,z) or viewing direction (θ,φ)
        out_dim = 0

        # Option to include the original input along with the encoded version
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # L-1, where L is number of frequency bands
        n_freqs = self.kwargs['num_freqs']       # L, number of frequency bands

        # Create frequency bands: 2^0, 2^1, 2^2, ..., 2^(L-1)
        # Log sampling means we sample frequencies logarithmically
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=n_freqs)

        # For each frequency band, apply both sin and cos
        # This creates: [sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ...]
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # [sin, cos]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d  # Each periodic function adds d dimensions

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # Apply all embedding functions and concatenate results
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
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
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,  # L-1
                'num_freqs' : multires,        # L
                'log_sampling' : True,         # Use logarithmic frequency sampling
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) MLP architecture.

    This implements the network F_Θ described in Section 3 of the paper.
    The network takes as input a 5D coordinate (position x,y,z and viewing direction θ,φ)
    and outputs volume density σ and RGB color c.

    Architecture details from paper (Section 3, Figure 3):
    - 8 fully-connected layers (D=8), 256 channels per layer (W=256)
    - Skip connection at layer 5 (concatenates input with intermediate features)
    - Position encoding applied separately to (x,y,z) and (θ,φ)
    - Density σ depends only on position (x,y,z)
    - RGB color c depends on both position and viewing direction
    """
    def __init__(self, depth=8, w=256, input_ch=3, input_ch_views=3, output_ch=4, skips=None, use_viewdirs=False):
        """
        Args:
            depth: Number of layers in the main MLP
            w: Width (number of channels) of each layer
            input_ch: Number of input channels for position (63 with positional encoding, L=10)
            input_ch_views: Number of input channels for viewing direction (27 with encoding, L=4)
            output_ch: Number of output channels (4 for RGB+density, or 5 for coarse/fine models)
            skips: Layers at which to add skip connections (typically [4] for layer 5)
            use_viewdirs: Whether to use viewing direction as input (enables view-dependent effects)
        """
        super(NeRF, self).__init__()
        if skips is None:
            skips = [4]
        self.D = depth
        self.W = w
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Main MLP for processing position
        # Consists of D layers with skip connections at specified layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, w)] + [nn.Linear(w, w) if i not in self.skips else nn.Linear(w + input_ch, w) for i in range(depth-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # Additional MLP for processing viewing direction (single layer in official implementation)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + w, w//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + w, w//2)] + [nn.Linear(w//2, w//2) for i in range(D//2)])

        if use_viewdirs:
            # When using viewing directions, split the network:
            # - alpha (density σ) depends only on position
            # - rgb (color c) depends on position and viewing direction
            self.feature_linear = nn.Linear(w, w)
            self.alpha_linear = nn.Linear(w, 1)       # Outputs volume density σ
            self.rgb_linear = nn.Linear(w//2, 3)      # Outputs RGB color c
        else:
            # Simple case: directly output RGB+density from position
            self.output_linear = nn.Linear(w, output_ch)

    def forward(self, x):
        """
        Forward pass through the NeRF network.

        Input format: concatenated [positionally_encoded_position, positionally_encoded_viewing_direction]

        Returns:
            outputs: [batch, 4] tensor containing [R, G, B, σ] where σ is volume density
        """
        # Split input into position and viewing direction components
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # Process through main MLP layers with skip connections
        # Skip connections help the network learn high-frequency details
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # Concatenate original input at skip layer (typically layer 5)
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Separate path for density (view-independent) and color (view-dependent)
            # This is key to modeling view-dependent effects like specularities
            alpha = self.alpha_linear(h)           # Volume density σ (view-independent)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)  # Concatenate viewing direction

            # Process through view-dependent layers
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)               # RGB color c (view-dependent)
            outputs = torch.cat([rgb, alpha], -1)  # [R, G, B, σ]
        else:
            # Simple case: both RGB and density from position only
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(h, w, k, c2w):
    """
    Generate ray origins and directions for all pixels in an image.

    This function implements the camera model to cast rays through each pixel.
    Rays are defined parametrically as: r(t) = o + td, where:
    - o is the ray origin (camera center)
    - d is the ray direction (unit vector)
    - t is the distance along the ray

    Args:
        h, w: Image height and width in pixels
        k: Camera intrinsic matrix [3x3] containing focal length and principal point
        c2w: Camera-to-world transformation matrix [3x4] (extrinsics)

    Returns:
        rays_o: [h, w, 3] Ray origins (all equal to camera center in world coordinates)
        rays_d: [h, w, 3] Ray directions in world coordinates
    """
    # Create pixel coordinate grid
    i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
    i = i.t()  # Transpose to get correct [h, W] shape
    j = j.t()

    # Convert pixel coordinates to normalized camera coordinates using intrinsics
    # k[0][0] = focal_x, k[1][1] = focal_y, k[0][2] = cx, k[1][2] = cy
    # This gives us ray directions in the camera coordinate system
    dirs = torch.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(h, w, k, c2w):
    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-k[0][2])/k[0][0], -(j-k[1][2])/k[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(h, w, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(w/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(h/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(w/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(h/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, n_samples, det=False, pytest=False):
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
        n_samples: Number of new samples to draw
        det: If True, use deterministic sampling; if False, use random sampling
        pytest: If True, use fixed random seed for reproducibility

    Returns:
        samples: [N_rays, n_samples] New sample locations along each ray
    """
    # Get pdf
    weights = weights + 1e-5 # prevent nans and ensure all weights are positive
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # Normalize to get probability distribution
    cdf = torch.cumsum(pdf, -1)                           # Cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples in [0, 1]
    if det:
        # Deterministic: evenly spaced samples
        u = torch.linspace(0., 1., steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        # Stochastic: random samples
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        rng = np.random.default_rng(0)
        new_shape = list(cdf.shape[:-1]) + [n_samples]
        if det:
            u = np.linspace(0., 1., n_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = rng.random(new_shape)
        u = torch.Tensor(u)

    # Invert CDF using binary search
    # For each uniform sample u, find where it falls in the CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # Gather the CDF and bin values at the indices
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Linear interpolation between the two surrounding bin values
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
