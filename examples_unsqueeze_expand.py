"""
Deep-dive: unsqueeze + expand in sample_pdf
===========================================

The goal of steps 8-9 is to answer, for every (ray, sample) pair:
  "What are the CDF values at the LEFT and RIGHT edges of this sample's bracket?"

We do that with torch.gather, but gather has a strict shape requirement.
This file shows exactly why the shapes don't match to start with, what each
operation does to fix that, and what the final gather produces.
"""

import torch

# ── Starting point ──────────────────────────────────────────────────────────
# After steps 1-3 we have the CDF:  shape (N_rays, N_bins+1) = (2, 5)
#   Row 0 (Ray 0): density peaked at bin 1  → CDF rises steeply early
#   Row 1 (Ray 1): density peaked at bin 0  → CDF rises steeply at the start
cdf = torch.tensor([
    [0.00, 0.10, 0.70, 0.90, 1.00],   # Ray 0
    [0.00, 0.70, 0.80, 0.90, 1.00],   # Ray 1
])
print("cdf shape:", cdf.shape)   # (2, 5)
print("cdf:\n", cdf)

# After step 7 we have the bracket index pairs:  shape (N_rays, N_samples, 2) = (2, 3, 2)
#   For each (ray, sample): [left_cdf_index, right_cdf_index]
inds_g = torch.tensor([
    [[0, 1], [1, 2], [3, 4]],   # Ray 0: 3 samples, each with a left+right bracket
    [[0, 1], [0, 1], [1, 2]],   # Ray 1: samples cluster near the early bins
])
print("\ninds_g shape:", inds_g.shape)   # (2, 3, 2)
print("inds_g:\n", inds_g)

# ── The problem ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("THE PROBLEM: dimension mismatch")
print("=" * 60)
print(f"  cdf    is {cdf.shape}   — 2D: (rays, bins)")
print(f"  inds_g is {inds_g.shape} — 3D: (rays, samples, 2-bracket-edges)")
print()
print("torch.gather requires BOTH tensors to have the same rank (ndim).")
print("We cannot call gather(cdf, 2, inds_g) because cdf has 2 dims")
print("but inds_g has 3.  We need to lift cdf to 3D.")
try:
    torch.gather(cdf, 2, inds_g)
except (IndexError, RuntimeError) as e:
    print(f"\nConfirmed — gather raises: {e}")

# ── Step 1: unsqueeze(1)  ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: cdf.unsqueeze(1)  —  insert a new 'samples' dimension")
print("=" * 60)
# unsqueeze(1) inserts a size-1 dimension at position 1.
# Think of it as: for each ray, wrap its CDF row in an extra layer.
cdf_unsqueezed = cdf.unsqueeze(1)
print(f"cdf shape before: {cdf.shape}")        # (2, 5)
print(f"cdf shape after:  {cdf_unsqueezed.shape}")   # (2, 1, 5)
print("cdf_unsqueezed:\n", cdf_unsqueezed)
print()
print("Meaning of each dimension now:")
print("  dim 0  → ray         (size 2)")
print("  dim 1  → samples     (size 1  ← placeholder, not yet N_samples)")
print("  dim 2  → bin values  (size 5)")

# ── Step 2: expand(matched_shape)  ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: .expand(2, 3, 5)  —  repeat the CDF row for each sample")
print("=" * 60)
# expand() stretches size-1 dimensions to a larger size WITHOUT copying data.
# dim 1 goes from 1 → 3 (N_samples).
# This is valid because dim 1 is exactly size 1 — the rule for expand.
matched_shape = [2, 3, 5]
cdf_expanded = cdf_unsqueezed.expand(matched_shape)
print(f"shape after expand: {cdf_expanded.shape}")   # (2, 3, 5)
print("cdf_expanded:\n", cdf_expanded)
print()
print("Notice: all 3 'sample' slices along dim-1 are IDENTICAL for each ray.")
print("Ray 0, sample 0:", cdf_expanded[0, 0])
print("Ray 0, sample 1:", cdf_expanded[0, 1])
print("Ray 0, sample 2:", cdf_expanded[0, 2])
print("  → same row, because every sample for the same ray uses the same CDF.")

# ── Why not just repeat() or tile?  ─────────────────────────────────────────
print("\n" + "=" * 60)
print("WHY expand AND NOT repeat/tile?")
print("=" * 60)
print("expand() is a zero-copy view — it just changes the stride metadata.")
print("repeat() would allocate N_samples copies of the data in memory.")
print("For a real scene: N_rays=1024, N_samples=128, N_bins=64")
print("  expand: no extra memory")
print("  repeat: 1024 * 128 * 64 * 4 bytes ≈ 33 MB of duplicated floats")

# ── Now gather works  ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: torch.gather — finally pick the bracket values")
print("=" * 60)
# gather(source, dim, index):
#   For each position (i,j,k) in inds_g, the output is:
#     out[i, j, k] = cdf_expanded[i, j, inds_g[i, j, k]]
#
# In plain English:
#   "For ray i, sample j, bracket edge k, look up column inds_g[i,j,k]
#    in that ray's CDF row."
cdf_g = torch.gather(cdf_expanded, 2, inds_g)
print(f"cdf_g shape: {cdf_g.shape}")   # (2, 3, 2)
print("cdf_g:\n", cdf_g)
print()
print("Reading the results — Ray 0:")
for s in range(3):
    left_idx  = inds_g[0, s, 0].item()
    right_idx = inds_g[0, s, 1].item()
    left_val  = cdf_g[0, s, 0].item()
    right_val = cdf_g[0, s, 1].item()
    print(f"  sample {s}: bracket [{left_idx}, {right_idx}]"
          f"  →  CDF values [{left_val:.2f}, {right_val:.2f}]"
          f"  (from cdf[0] = {cdf[0].tolist()})")

# ── Shape journey summary  ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SHAPE JOURNEY SUMMARY")
print("=" * 60)
print(f"  cdf              {(2,5)}   — one CDF row per ray")
print(f"  .unsqueeze(1) →  {(2,1,5)} — add placeholder sample dim")
print(f"  .expand(...) →   {(2,3,5)} — repeat (virtually) for each sample")
print(f"  .gather(2,inds)→ {(2,3,2)} — pick left+right bracket values")
print()
print("The (2,3,2) result feeds directly into the linear interpolation:")
print("  cdf_g[...,0] = left  CDF value of each bracket")
print("  cdf_g[...,1] = right CDF value of each bracket")
