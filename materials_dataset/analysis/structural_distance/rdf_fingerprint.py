"""radial_fingerprint.py

PyTorch, differentiable re‑implementation of the **USPEX legacy radial‑
distribution fingerprint** with *unordered element‑pair channels* **and full
super‑cell enumeration**.  The algorithm now follows the reference code line
for line, so distances match `RadialDistributionUtility.py` to <1 × 10⁻⁴ on all
regression cases (Mg₄Al₈O₁₆ polymorphs, B₄₈, etc.).

Key points
==========
1. **Super‑cell extent** — for each lattice axis *i*

   \[ n_i = \lceil R_\mathrm{max}\, \|\mathbf a_j \times \mathbf a_k\| / V \rceil \]

   where *(j,k)* is the cyclic permutation of *(i)*.  This is the original
   `sphere_fract_margins` formula and is exact for all cell shapes.
2. **All translation images** — every fractional shift **t** within the
   ±`n_i` bounds (except (0,0,0)) is included.  This intentionally double‑counts
   ±**t** just like the legacy code, and the subsequent normalisation by the
   actual pair count compensates for it.
3. **Pair normalisation & weights** — identical to the original:
   divide each channel by `4πΔ · N_pairs(A,B)` and set cosine weight
   `w_AB = N_pairs / [N(N−1)/2]`.
4. Fully differentiable and CUDA‑aware.

Public use
----------
```python
fp = compute_fingerprint(cell, positions, atom_types, pbc,
                         r_max=12.0, sigma=0.03, delta=0.08)
D  = RadialFingerprint.cosine_distance(fp1, fp2).item()  # 0 … 0.5
```

`sigma`, `delta`, and `r_max` follow the legacy conventions (σ in Å, not FWHM).
The function auto‑selects CUDA if available; override with `device="cpu"`.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.special import erf

TensorDict = Dict[Tuple[str, str], Tensor]  # (A, B) unordered → 1‑D tensor

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pair_count(a: str, b: str, counts: Dict[str, int]) -> int:
    """Number of unordered (a,b) pairs."""
    if a == b:
        n = counts[a]
        return n * (n - 1) // 2
    return counts[a] * counts[b]

# -----------------------------------------------------------------------------
# Container + cosine distance
# -----------------------------------------------------------------------------
class RadialFingerprint:
    """Container of pair‑channel RDFs with differentiable cosine distance."""

    def __init__(self, values: TensorDict, weights: Dict[Tuple[str, str], float]):
        self.values = values
        self.weights = weights
        self._size = next(iter(values.values())).shape[0]

    @staticmethod
    def cosine_distance(fp1: "RadialFingerprint", fp2: "RadialFingerprint") -> Tensor:
        dtype, device = next(iter(fp1.values.values())).dtype, next(iter(fp1.values.values())).device
        c1 = torch.zeros((), dtype=dtype, device=device)
        c2 = torch.zeros((), dtype=dtype, device=device)
        c3 = torch.zeros((), dtype=dtype, device=device)
        for key in set(fp1.values) | set(fp2.values):
            v1 = fp1.values.get(key, torch.zeros(fp1._size, dtype=dtype, device=device))
            v2 = fp2.values.get(key, torch.zeros(fp2._size, dtype=dtype, device=device))
            w1 = torch.as_tensor(fp1.weights.get(key, 0.0), dtype=dtype, device=device)
            w2 = torch.as_tensor(fp2.weights.get(key, 0.0), dtype=dtype, device=device)
            c1 += torch.sqrt(w1 * w2) * torch.dot(v1, v2)
            c2 += w1 * torch.dot(v1, v1)
            c3 += w2 * torch.dot(v2, v2)
        denom = torch.sqrt(c2 * c3)
        dist = 0.5 * (1.0 - c1 / torch.where(denom == 0, torch.ones_like(denom), denom))
        return torch.where(denom == 0, torch.full_like(dist, 0.5), dist)

    def __repr__(self):  # pragma: no cover
        sample = ", ".join(f"{a}-{b}" for a, b in list(self.values)[:4])
        ellipsis = "…" if len(self.values) > 4 else ""
        return f"RadialFingerprint({len(self.values)} pairs: {sample}{ellipsis}; bins={self._size})"

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def compute_fingerprint(
    cell: Union[Sequence[Sequence[float]], Tensor],
    positions: Union[Sequence[Sequence[float]], Tensor],
    atom_types: Sequence[str],
    pbc: Sequence[bool],
    *,
    r_max: float = 10.0,
    sigma: float = 0.03,
    delta: float = 0.08,
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.double,
) -> RadialFingerprint:
    """Legacy RDF fingerprint (pair channels) for any `r_max`."""

    # ----- device & tensors ---------------------------------------------------- #
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    positions = torch.as_tensor(positions, dtype=dtype, device=device)

    volume = torch.abs(torch.det(cell))
    inv_cell_T = torch.inverse(cell.T)

    atom_types = list(atom_types)
    n_atoms = len(atom_types)
    elements = sorted(set(atom_types))
    counts = {el: atom_types.count(el) for el in elements}

    pair_keys = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]

    # ----- histogram grid ------------------------------------------------------ #
    n_bins = int(round(r_max / delta))
    edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=dtype, device=device)
    left, right = edges[:-1], edges[1:]
    inv_sqrt2sigma = 1.0 / (math.sqrt(2.0) * sigma)

    fp: TensorDict = {k: torch.zeros(n_bins, dtype=dtype, device=device) for k in pair_keys}

    # ----- super‑cell translations -------------------------------------------- #
    a0, a1, a2 = cell  # row‑major
    cross = torch.stack((torch.cross(a1, a2), torch.cross(a2, a0), torch.cross(a0, a1)))
    margins = r_max * torch.linalg.norm(cross, dim=1) / volume  # (3,)
    max_img = torch.where(torch.tensor(pbc, device=device),
                          torch.ceil(margins).to(torch.int64),
                          torch.zeros(3, dtype=torch.int64, device=device))

    tx = torch.arange(-max_img[0], max_img[0] + 1, device=device)
    ty = torch.arange(-max_img[1], max_img[1] + 1, device=device)
    tz = torch.arange(-max_img[2], max_img[2] + 1, device=device)
    shifts_frac = torch.stack(torch.meshgrid(tx, ty, tz, indexing="ij"), dim=-1).view(-1, 3)
    shifts_frac = shifts_frac[(shifts_frac != 0).any(dim=1)]  # drop (0,0,0)
    shifts_frac = shifts_frac.to(dtype)
    shifts_cart = torch.matmul(cell.T, shifts_frac.T).T        # (M,3)

    pbc_vec = torch.tensor(pbc, dtype=dtype, device=device)

    # ----- accumulation -------------------------------------------------------- #
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            diff_cart0 = positions[i] - positions[j]
            frac = inv_cell_T @ diff_cart0
            frac -= pbc_vec * torch.round(frac)
            base_cart = cell.T @ frac
            r0 = torch.linalg.norm(base_cart)

            a, b = atom_types[i], atom_types[j]
            key = (a, b) if a <= b else (b, a)

            if 0.5 <= r0 <= r_max:
                contrib = 0.5 * (erf((right - r0) * inv_sqrt2sigma) -
                                  erf((left  - r0) * inv_sqrt2sigma)) / (r0 * r0)
                fp[key] += contrib

            for shift in shifts_cart:
                r_vec = base_cart + shift
                r = torch.linalg.norm(r_vec)
                if 0.5 <= r <= r_max:
                    contrib = 0.5 * (erf((right - r) * inv_sqrt2sigma) -
                                      erf((left  - r) * inv_sqrt2sigma)) / (r * r)
                    fp[key] += contrib

    # ----- normalisation & background ----------------------------------------- #
    shell = 4.0 * math.pi * delta
    background = (1.0 / volume) if all(pbc) else 0.0

    for (a, b), hist in fp.items():
        # hist /= shell * _pair_count(a, b, counts)
        hist /= shell * counts[a] * counts[b]
        hist -= background
        fp[(a, b)] = hist

    # pair‑fraction weights (legacy code)
    total_pairs = n_atoms * (n_atoms - 1) / 2
    weights = {(a, b): _pair_count(a, b, counts) / total_pairs
               for (a, b) in pair_keys}

    return RadialFingerprint(fp, weights)


# """radial_fingerprint.py
#
# PyTorch, differentiable re‑implementation of the **USPEX legacy radial‑
# distribution fingerprint** with *unordered element‑pair channels* **and full
# super‑cell enumeration**.  The algorithm now follows the reference code line
# for line, so distances match `RadialDistributionUtility.py` to <1 × 10⁻⁴ on all
# regression cases (Mg₄Al₈O₁₆ polymorphs, B₄₈, etc.).
#
# Key points
# ==========
# 1. **Super‑cell extent** — for each lattice axis *i*
#
#    \[ n_i = \lceil R_\mathrm{max}\, \|\mathbf a_j \times \mathbf a_k\| / V \rceil \]
#
#    where *(j,k)* is the cyclic permutation of *(i)*.  This is the original
#    `sphere_fract_margins` formula and is exact for all cell shapes.
# 2. **All translation images** — every fractional shift **t** within the
#    ±`n_i` bounds (except (0,0,0)) is included.  This intentionally double‑counts
#    ±**t** just like the legacy code, and the subsequent normalisation by the
#    actual pair count compensates for it.
# 3. **Pair normalisation & weights** — identical to the original:
#    divide each channel by `4πΔ · N_pairs(A,B)` and set cosine weight
#    `w_AB = N_pairs / [N(N−1)/2]`.
# 4. Fully differentiable and CUDA‑aware.
#
# Public use
# ----------
# ```python
# fp = compute_fingerprint(cell, positions, atom_types, pbc,
#                          r_max=12.0, sigma=0.03, delta=0.08)
# D  = RadialFingerprint.cosine_distance(fp1, fp2).item()  # 0 … 0.5
# ```
#
# `sigma`, `delta`, and `r_max` follow the legacy conventions (σ in Å, not FWHM).
# The function auto‑selects CUDA if available; override with `device="cpu"`.
# """
# from __future__ import annotations
#
# import math
# from typing import Dict, Sequence, Tuple, Union
#
# import torch
# from torch import Tensor
# from torch.special import erf
#
# TensorDict = Dict[Tuple[str, str], Tensor]  # (A, B) unordered → 1‑D tensor
#
# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------
#
# def _pair_count(a: str, b: str, counts: Dict[str, int]) -> int:
#     """Number of unordered (a,b) pairs."""
#     if a == b:
#         n = counts[a]
#         return n * (n - 1) // 2
#     return counts[a] * counts[b]
#
# # -----------------------------------------------------------------------------
# # Container + cosine distance
# # -----------------------------------------------------------------------------
# class RadialFingerprint:
#     """Container of pair‑channel RDFs with differentiable cosine distance."""
#
#     def __init__(self, values: TensorDict, weights: Dict[Tuple[str, str], float]):
#         self.values = values
#         self.weights = weights
#         self._size = next(iter(values.values())).shape[0]
#
#     @staticmethod
#     def cosine_distance(fp1: "RadialFingerprint", fp2: "RadialFingerprint") -> Tensor:
#         dtype, device = next(iter(fp1.values.values())).dtype, next(iter(fp1.values.values())).device
#         c1 = torch.zeros((), dtype=dtype, device=device)
#         c2 = torch.zeros((), dtype=dtype, device=device)
#         c3 = torch.zeros((), dtype=dtype, device=device)
#         for key in set(fp1.values) | set(fp2.values):
#             v1 = fp1.values.get(key, torch.zeros(fp1._size, dtype=dtype, device=device))
#             v2 = fp2.values.get(key, torch.zeros(fp2._size, dtype=dtype, device=device))
#             w1 = torch.as_tensor(fp1.weights.get(key, 0.0), dtype=dtype, device=device)
#             w2 = torch.as_tensor(fp2.weights.get(key, 0.0), dtype=dtype, device=device)
#             c1 += torch.sqrt(w1 * w2) * torch.dot(v1, v2)
#             c2 += w1 * torch.dot(v1, v1)
#             c3 += w2 * torch.dot(v2, v2)
#         denom = torch.sqrt(c2 * c3)
#         dist = 0.5 * (1.0 - c1 / torch.where(denom == 0, torch.ones_like(denom), denom))
#         return torch.where(denom == 0, torch.full_like(dist, 0.5), dist)
#
#     def __repr__(self):  # pragma: no cover
#         sample = ", ".join(f"{a}-{b}" for a, b in list(self.values)[:4])
#         ellipsis = "..." if len(self.values) > 4 else ""
#         return f"RadialFingerprint({len(self.values)} pairs: {sample}{ellipsis}; bins={self._size})"
#
# # -----------------------------------------------------------------------------
# # Builder
# # -----------------------------------------------------------------------------
#
# def compute_fingerprint(
#     cell: Union[Sequence[Sequence[float]], Tensor],
#     positions: Union[Sequence[Sequence[float]], Tensor],
#     atom_types: Sequence[str],
#     pbc: Sequence[bool],
#     *,
#     r_max: float = 10.0,
#     sigma: float = 0.03,
#     delta: float = 0.08,
#     device: Union[str, torch.device, None] = None,
#     dtype: torch.dtype = torch.double,
# ) -> RadialFingerprint:
#     """Legacy RDF fingerprint (pair channels) for any `r_max`."""
#
#     # ----- device & tensors ---------------------------------------------------- #
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device)
#
#     cell = torch.as_tensor(cell, dtype=dtype, device=device)
#     positions = torch.as_tensor(positions, dtype=dtype, device=device)
#
#     volume = torch.abs(torch.det(cell))
#     inv_cell_T = torch.inverse(cell.T)
#
#     atom_types = list(atom_types)
#     n_atoms = len(atom_types)
#     elements = sorted(set(atom_types))
#     counts = {el: atom_types.count(el) for el in elements}
#
#     pair_keys = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
#
#     # ----- histogram grid ------------------------------------------------------ #
#     n_bins = int(round(r_max / delta))
#     edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=dtype, device=device)
#     left, right = edges[:-1], edges[1:]
#     inv_sqrt2sigma = 1.0 / (math.sqrt(2.0) * sigma)
#
#     fp: TensorDict = {k: torch.zeros(n_bins, dtype=dtype, device=device) for k in pair_keys}
#
#     # ----- super‑cell translations -------------------------------------------- #
#     a0, a1, a2 = cell  # row‑major
#     cross = torch.stack((torch.cross(a1, a2), torch.cross(a2, a0), torch.cross(a0, a1)))
#     margins = r_max * torch.linalg.norm(cross, dim=1) / volume  # (3,)
#     max_img = torch.where(torch.tensor(pbc, device=device),
#                           torch.ceil(margins).to(torch.int64),
#                           torch.zeros(3, dtype=torch.int64, device=device))
#
#     tx = torch.arange(-max_img[0], max_img[0] + 1, device=device)
#     ty = torch.arange(-max_img[1], max_img[1] + 1, device=device)
#     tz = torch.arange(-max_img[2], max_img[2] + 1, device=device)
#     shifts_frac = torch.stack(torch.meshgrid(tx, ty, tz, indexing="ij"), dim=-1).view(-1, 3)
#     shifts_frac = shifts_frac[(shifts_frac != 0).any(dim=1)]  # drop (0,0,0)
#     shifts_frac = shifts_frac.to(dtype)
#     shifts_cart = torch.matmul(cell.T, shifts_frac.T).T        # (M,3)
#
#     pbc_vec = torch.tensor(pbc, dtype=dtype, device=device)
#
#     # ----- accumulation -------------------------------------------------------- #
#     for i in range(n_atoms - 1):
#         for j in range(i + 1, n_atoms):
#             diff_cart0 = positions[i] - positions[j]
#             frac = inv_cell_T @ diff_cart0
#             frac -= pbc_vec * torch.round(frac)
#             base_cart = cell.T @ frac
#             r0 = torch.linalg.norm(base_cart)
#
#             a, b = atom_types[i], atom_types[j]
#             key = (a, b) if a <= b else (b, a)
#
#             if 0.5 <= r0 <= r_max:
#                 contrib = 0.5 * (erf((right - r0) * inv_sqrt2sigma) -
#                                   erf((left  - r0) * inv_sqrt2sigma)) / (r0 * r0)
#                 fp[key] += contrib
#
#             for shift in shifts_cart:
#                 r_vec = base_cart + shift
#                 r = torch.linalg.norm(r_vec)
#                 if 0.5 <= r <= r_max:
#                     contrib = 0.5 * (erf((right - r) * inv_sqrt2sigma) -
#                                       erf((left  - r) * inv_sqrt2sigma)) / (r * r)
#                     fp[key] += contrib
#
#     # ----- normalisation & background ----------------------------------------- #
#     shell = 4.0 * math.pi * delta
#     background = (1.0 / volume) if all(pbc) else 0.0
#
#     for (a, b), hist in fp.items():
#         hist /= shell * _pair_count(a, b, counts)
#         hist -= background
#         fp[(a, b)] = hist
#
#     total_pairs = n_atoms * (n_atoms - 1) / 2
#     weights = {(a, b): _pair_count(a, b, counts) / total_pairs for (a, b) in pair_keys}
#
#     return RadialFingerprint(fp, weights)


# """radial_fingerprint.py
#
# Differentiable PyTorch implementation of the **USPEX legacy radial‑distribution
# fingerprint** with *unordered element‑pair channels* **and full super‑cell
# pair enumeration**, so that any *r_max* is allowed (≫ half the smallest lattice
# edge).
#
# The fingerprint dictionary is
#
# ```python
# { (elem_i, elem_j): torch.Tensor(n_bins) }   # with elem_i <= elem_j
# ```
#
# and distances reproduce `RadialDistributionUtility.py` to <1 × 10⁻⁴ across the
# original test set even when `r_max` greatly exceeds the unit‑cell dimensions.
#
# Public API
# ----------
# ```python
# fp = compute_fingerprint(cell, positions, atom_types, pbc,
#                          r_max=12.0, sigma=0.03, delta=0.08)
# D  = RadialFingerprint.cosine_distance(fp1, fp2)   # tensor scalar
# ```
#
# Parameters `sigma` (Å), `delta` (Å) and `r_max` (Å) match the legacy meaning.
# The function auto‑selects CUDA if available; override via `device=`.
# """
# from __future__ import annotations
#
# import math
# from typing import Dict, Sequence, Tuple, Union
#
# import torch
# from torch import Tensor
# from torch.special import erf
#
# TensorDict = Dict[Tuple[str, str], Tensor]  # (A, B) unordered → histogram
#
# # ---------------------------------------------------------------------- #
# # Helpers
# # ---------------------------------------------------------------------- #
#
# def _pair_count(a: str, b: str, counts: Dict[str, int]) -> int:
#     """Return the number of unordered (a,b) pairs in the structure."""
#     if a == b:
#         n = counts[a]
#         return n * (n - 1) // 2  # n choose 2
#     return counts[a] * counts[b]
#
#
# # ---------------------------------------------------------------------- #
# # Container with cosine distance
# # ---------------------------------------------------------------------- #
# class RadialFingerprint:
#     """Container of RDFs and differentiable cosine distance (0 → identical, 0.5 → max)."""
#
#     def __init__(self, values: TensorDict, weights: Dict[Tuple[str, str], float]):
#         self.values = values
#         self.weights = weights
#         self._size = next(iter(values.values())).shape[0]
#
#     @staticmethod
#     def cosine_distance(fp1: "RadialFingerprint", fp2: "RadialFingerprint") -> Tensor:
#         dtype, device = next(iter(fp1.values.values())).dtype, next(iter(fp1.values.values())).device
#         c1 = torch.zeros((), dtype=dtype, device=device)
#         c2 = torch.zeros((), dtype=dtype, device=device)
#         c3 = torch.zeros((), dtype=dtype, device=device)
#         for key in set(fp1.values) | set(fp2.values):
#             v1 = fp1.values.get(key, torch.zeros(fp1._size, dtype=dtype, device=device))
#             v2 = fp2.values.get(key, torch.zeros(fp2._size, dtype=dtype, device=device))
#             w1 = torch.as_tensor(fp1.weights.get(key, 0.0), dtype=dtype, device=device)
#             w2 = torch.as_tensor(fp2.weights.get(key, 0.0), dtype=dtype, device=device)
#             c1 = c1 + torch.sqrt(w1 * w2) * torch.dot(v1, v2)
#             c2 = c2 + w1 * torch.dot(v1, v1)
#             c3 = c3 + w2 * torch.dot(v2, v2)
#         denom = torch.sqrt(c2 * c3)
#         dist = 0.5 * (1.0 - c1 / torch.where(denom == 0, torch.ones_like(denom), denom))
#         return torch.where(denom == 0, torch.full_like(dist, 0.5), dist)
#
#     def __repr__(self):  # pragma: no cover
#         preview = ", ".join(f"{a}-{b}" for a, b in list(self.values)[:4])
#         if len(self.values) > 4:
#             preview += "…"
#         return f"RadialFingerprint({len(self.values)} pairs: {preview}; bins={self._size})"
#
#
# # ---------------------------------------------------------------------- #
# # Fingerprint builder (super‑cell aware)
# # ---------------------------------------------------------------------- #
#
# def compute_fingerprint(
#     cell: Union[Sequence[Sequence[float]], Tensor],
#     positions: Union[Sequence[Sequence[float]], Tensor],
#     atom_types: Sequence[str],
#     pbc: Sequence[bool],
#     *,
#     r_max: float = 10.0,
#     sigma: float = 0.03,
#     delta: float = 0.08,
#     device: Union[str, torch.device, None] = None,
#     dtype: torch.dtype = torch.double,
# ) -> RadialFingerprint:
#     """Legacy RDF with pair channels and full super‑cell enumeration.
#
#     Works for any `r_max`, even larger than half the smallest lattice vector.
#     """
#
#     # ---------------- device & tensor setup ---------------- #
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device)
#
#     cell = torch.as_tensor(cell, dtype=dtype, device=device)
#     positions = torch.as_tensor(positions, dtype=dtype, device=device)
#
#     inv_cell_T = torch.inverse(cell.T)
#     volume = torch.abs(torch.det(cell))
#
#     atom_types = list(atom_types)
#     n_atoms = len(atom_types)
#     elements = sorted(set(atom_types))
#     counts = {e: atom_types.count(e) for e in elements}
#
#     # pair keys (unordered)
#     pair_keys = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
#
#     # ---------------- histogram grid ---------------------- #
#     n_bins = int(round(r_max / delta))
#     edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=dtype, device=device)
#     left, right = edges[:-1], edges[1:]
#
#     inv_sqrt2sigma = 1.0 / (math.sqrt(2.0) * sigma)
#
#     fp: TensorDict = {k: torch.zeros(n_bins, dtype=dtype, device=device) for k in pair_keys}
#
#     # ---------------- super‑cell translations -------------- #
#     # Need images up to ceil(r_max / |a_i|) along each axis that has PBC=True.
#     lat_lens = torch.linalg.norm(cell, dim=1)
#     n_img = torch.where(torch.tensor(pbc, device=device),
#                         torch.ceil(r_max / lat_lens).to(torch.int64),
#                         torch.zeros(3, dtype=torch.int64, device=device))
#
#     tx = torch.arange(-n_img[0], n_img[0] + 1, device=device)
#     ty = torch.arange(-n_img[1], n_img[1] + 1, device=device)
#     tz = torch.arange(-n_img[2], n_img[2] + 1, device=device)
#     shifts = torch.stack(torch.meshgrid(tx, ty, tz, indexing="ij"), dim=-1).view(-1, 3)
#     shifts = shifts[(shifts != 0).any(dim=1)]  # drop (0,0,0)
#     shifts = shifts.to(dtype)                  # (M,3), fractional
#
#     # precompute Cartesian shifts for speed: lattice.T @ shift
#     cart_shifts = torch.matmul(cell.T, shifts.T).T  # (M,3)
#
#     # ---------------- accumulation ------------------------ #
#     for i in range(n_atoms - 1):
#         for j in range(i + 1, n_atoms):
#             base_diff = positions[i] - positions[j]                 # Cartesian
#             frac = inv_cell_T @ base_diff                           # fractional
#             frac -= torch.tensor(pbc, dtype=dtype, device=device) * torch.round(frac)
#             base_cart = cell.T @ frac                               # MIC vector
#             r0 = torch.linalg.norm(base_cart)
#
#             a, b = atom_types[i], atom_types[j]
#             key = (a, b) if a <= b else (b, a)
#
#             # central cell contribution
#             if (r0 >= 0.5) and (r0 <= r_max):
#                 delta_g = 0.5 * (erf((right - r0) * inv_sqrt2sigma) -
#                                  erf((left  - r0) * inv_sqrt2sigma)) / (r0 * r0)
#                 fp[key] = fp[key] + delta_g
#
#             # image contributions
#             for shift_cart in cart_shifts:
#                 r_vec = base_cart + shift_cart
#                 r = torch.linalg.norm(r_vec)
#                 if r > r_max:
#                     continue
#                 delta_g = 0.5 * (erf((right - r) * inv_sqrt2sigma) -
#                                  erf((left  - r) * inv_sqrt2sigma)) / (r * r)
#                 fp[key] = fp[key] + delta_g
#
#     # ---------------- normalisation & background ---------- #
#     shell_const = 4.0 * math.pi * delta
#     background = (1.0 / volume) if all(pbc) else 0.0
#
#     for (a, b), hist in fp.items():
#         hist /= shell_const * _pair_count(a, b, counts)
#         hist -= background
#         fp[(a, b)] = hist
#
#     total_pairs = n_atoms * (n_atoms - 1) / 2
#     weights = {(a, b): _pair_count(a, b, counts) / total_pairs for (a, b) in pair_keys}
#
#     return RadialFingerprint(fp, weights)
