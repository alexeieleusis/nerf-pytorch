"""
Illustrative examples of the PyTorch functions used in sample_pdf().

We use a concrete 2-ray, 4-bin scenario throughout so every tensor
has meaningful values you can reason about.

Scenario
--------
Imagine the coarse NeRF network evaluated 4 depth bins along 2 rays and
returned raw weights proportional to the density it found there.

    Ray 0: most density near bin 2  →  weights ≈ [0.1, 0.6, 0.2, 0.1]
    Ray 1: most density near bin 0  →  weights ≈ [0.7, 0.1, 0.1, 0.1]

We want the fine network to draw N_samples=3 new depth values per ray,
concentrating them where the coarse network found high density.
"""

import torch

print("=" * 60)
print("Setup: coarse weights per ray (2 rays, 4 bins)")
print("=" * 60)

# Raw weights from the coarse network (2 rays × 4 bins)
weights_raw = 10 * torch.tensor([
    [0.1, 0.6, 0.2, 0.1],  # Ray 0: peak at bin index 1
    [0.7, 0.1, 0.1, 0.1],  # Ray 1: peak at bin index 0
])
print("weights_raw:\n", weights_raw)

# In sample_pdf the first step is to add a small epsilon to avoid zero weights.
weights = weights_raw + 1e-5
print("\nweights (+ 1e-5 to avoid NaN):\n", weights)


# ── 1. torch.sum  ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. torch.sum – normalise weights into a valid PDF")
print("=" * 60)
# Sum across the bin dimension (dim=-1) so we get a total per ray.
# keepdim=True preserves the shape for broadcasting in the division below.
totals = torch.sum(weights, dim=-1, keepdim=True)
print("totals per ray (keepdim=True):", totals)  # shape (2, 1)

pdf = weights / totals
print("pdf (each row sums to 1):\n", pdf)
print("row sums:", pdf.sum(dim=-1))  # should both be 1.0


# ── 2. torch.cumsum  ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. torch.cumsum – build the CDF from the PDF")
print("=" * 60)
# cumsum adds each element to all previous elements → monotonically rising
# from ~0 to 1. It encodes "what fraction of the total weight is to the
# left of bin i".
cdf_no_zero = torch.cumsum(pdf, dim=-1)
print("CDF (without leading zero):\n", cdf_no_zero)
# Ray 0 example: [0.1, 0.7, 0.9, 1.0]  →  70 % of weight is in bins 0-1


# ── 3. torch.cat  ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. torch.cat – prepend a zero column so the CDF starts at 0")
print("=" * 60)
# We need the CDF to span [0, 1] so that any uniform sample u ∈ [0,1]
# has a valid left-bracket to land in.
zeros = torch.zeros_like(cdf_no_zero[..., :1])  # shape (2, 1), all zeros
print("zeros to prepend:\n", zeros)
cdf = torch.cat([zeros, cdf_no_zero], dim=-1)  # shape (2, 5)
print("CDF (with leading zero):\n", cdf)


# ── 4. torch.linspace / torch.rand  ────────────────────────────────────────
print("\n" + "=" * 60)
print("4. torch.linspace & torch.rand – draw uniform quantile samples")
print("=" * 60)
N_samples = 3

# Deterministic path (used at test/render time)
u_det = torch.linspace(0.0, 1.0, steps=N_samples)  # shape (N_samples,)
u_det = u_det.expand([2, N_samples])  # broadcast to (2, 3)
print("Deterministic u (evenly spaced):\n", u_det)

# Stochastic path (used during training)
torch.manual_seed(42)
u_rand = torch.rand([2, N_samples])
print("Random u:\n", u_rand)

# Use deterministic for the rest of this demo
u = u_det.contiguous()  # .contiguous() required by searchsorted


# ── 5. torch.searchsorted  ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. torch.searchsorted – invert the CDF (binary search)")
print("=" * 60)
# For each query u find the index i such that cdf[i-1] <= u < cdf[i].
# right=True: when u equals a CDF value exactly, we take the RIGHT bracket.
inds = torch.searchsorted(cdf, u, right=True)
print("CDF:\n", cdf)
print("Query u:\n", u)
print("Insertion indices:\n", inds)
# Ray 0, u=0.0 → index 1 (just after the prepended zero)
# Ray 0, u=0.5 → index 2 (lands in the big bin 1 region)
# Ray 0, u=1.0 → index 5 (at or beyond the last bin)


# ── 6. torch.max / torch.min  ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. torch.max / torch.min – clamp indices to valid range")
print("=" * 60)
# 'below' is the left edge of the bracket; 'above' is the right edge.
# Both must stay inside [0, len(cdf)-1] to avoid out-of-bounds gathers.
below = torch.max(torch.zeros_like(inds - 1), inds - 1)
above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
print("below (left  bracket index):\n", below)
print("above (right bracket index):\n", above)


# ── 7. torch.stack  ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. torch.stack – pair (below, above) into a single gather-index tensor")
print("=" * 60)
inds_g = torch.stack([below, above], dim=-1)  # shape (2, N_samples, 2)
print("inds_g shape:", inds_g.shape)  # (2, 3, 2)
print("inds_g:\n", inds_g)
# Each (ray, sample) row is [left_idx, right_idx] of the bracket.


# ── 8. tensor.unsqueeze / tensor.expand  ───────────────────────────────────
print("\n" + "=" * 60)
print("8. unsqueeze + expand – broadcast CDF/bins for batched gather")
print("=" * 60)
# The problem: torch.gather requires source and index tensors to have the
# same number of dimensions (rank), but right now they don't match:
#   cdf    is (2, 5)    — 2D: (rays, bins)
#   inds_g is (2, 3, 2) — 3D: (rays, samples, 2 bracket edges)
# We need to lift cdf from 2D to 3D before gather will accept it.
print("Dimension mismatch before fix:")
print(f"  cdf    shape: {cdf.shape}    — 2D: (rays, bins)")
print(f"  inds_g shape: {inds_g.shape} — 3D: (rays, samples, bracket_edges)")
try:
    torch.gather(cdf, 2, inds_g)
except (IndexError, RuntimeError) as e:
    print(f"  gather raises: {e}")

# -- Step 8a: unsqueeze(1) ---------------------------------------------------
# Insert a new size-1 dimension at position 1.
# (2, 5) → (2, 1, 5)
# The three dimensions now mean: (rays, samples_placeholder, bin_values).
# The '1' signals "this axis can be stretched" — it doesn't copy any data.
cdf_unsqueezed = cdf.unsqueeze(1)
print(f"\nAfter unsqueeze(1): {cdf.shape} → {cdf_unsqueezed.shape}")
print("  dim 0 → rays           (size 2)")
print("  dim 1 → samples        (size 1  ← placeholder)")
print("  dim 2 → bin values     (size 5)")
print("cdf_unsqueezed:\n", cdf_unsqueezed)

# -- Step 8b: expand(matched_shape) -----------------------------------------
# Stretch the size-1 placeholder dimension from 1 → N_samples (3).
# (2, 1, 5) → (2, 3, 5)
# expand() is a zero-copy view: it adjusts stride metadata so the same
# memory row appears N_samples times. No data is duplicated.
# (Using repeat() instead would physically allocate N_samples copies —
# for a real run: 1024 rays × 128 samples = ~33 MB of wasted allocation.)
matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
cdf_expanded = cdf_unsqueezed.expand(matched_shape)
print(f"\nAfter expand{matched_shape}: {cdf_unsqueezed.shape} → {cdf_expanded.shape}")
print("cdf_expanded:\n", cdf_expanded)
print("\nAll 3 sample slices for Ray 0 are the same CDF row (zero-copy):")
for s in range(3):
    print(f"  sample {s}: {cdf_expanded[0, s].tolist()}")


# ── 9. torch.gather  ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. torch.gather – collect CDF and bin values at bracket edges")
print("=" * 60)
# gather(source, dim, index):
#   For every position (i, j, k) in inds_g, the output is:
#     out[i, j, k] = cdf_expanded[i, j, inds_g[i, j, k]]
# In plain English: "for ray i, sample j, bracket edge k,
#   look up column inds_g[i,j,k] in that ray's CDF row."
# Shape journey:  (2,5) → unsqueeze → (2,1,5) → expand → (2,3,5) → gather → (2,3,2)
cdf_g = torch.gather(cdf_expanded, 2, inds_g)  # shape (2, N_samples, 2)
print("cdf_g (left and right CDF values per sample):\n", cdf_g)
print("\nReading the results — Ray 0:")
for s in range(3):
    li, ri = inds_g[0, s, 0].item(), inds_g[0, s, 1].item()
    lv, rv = cdf_g[0, s, 0].item(), cdf_g[0, s, 1].item()
    print(f"  sample {s}: bracket indices [{li}, {ri}]  →  CDF values [{lv:.3f}, {rv:.3f}]")

# Same for the bin depth positions
bins = torch.tensor([
    [0.0, 0.25, 0.50, 0.75, 1.0],  # 5 depth positions for Ray 0
    [0.0, 0.25, 0.50, 0.75, 1.0],  # 5 depth positions for Ray 1
])
bins_expanded = bins.unsqueeze(1).expand(matched_shape)
bins_g = torch.gather(bins_expanded, 2, inds_g)
print("bins_g (left and right depth values per sample):\n", bins_g)


# ── 10. torch.where  ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. torch.where – guard against zero-width CDF brackets")
print("=" * 60)
# If two adjacent CDF values are identical (e.g. a zero-weight bin), the
# denominator would be 0, giving NaN. We replace those with 1 so t becomes
# 0 and we just return the left bin edge (a safe fallback).
denom = cdf_g[..., 1] - cdf_g[..., 0]
print("raw denom:\n", denom)

denom_safe = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
print("safe denom (zeros replaced with 1):\n", denom_safe)


# ── Final interpolation ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final: linear interpolation → new sample depths")
print("=" * 60)
t = (u - cdf_g[..., 0]) / denom_safe  # fractional position in bracket
samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
print("t (fraction within bracket):\n", t)
print("samples (new depth locations per ray):\n", samples)
print("\nInterpretation:")
print("  Ray 0: samples concentrated around the high-density region (bins 1-2)")
print("  Ray 1: samples concentrated around the high-density region (bin 0)")
