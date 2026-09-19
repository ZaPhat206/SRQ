"""Fixed feature maps: FLY-CL winner-take-all codes, RanPAC random ReLU, CountSketch."""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# FLY-CL: Phi(x) = WTA_rho(P x) with a sparse Gaussian projection P
# ---------------------------------------------------------------------------
def fly_projection(feature_dim: int, width: int, synaptic_degree: int, seed: int,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    """Sparse CSC projection of shape ``(width, feature_dim)``.

    Every row has ``synaptic_degree`` standard-normal entries at columns drawn
    without replacement.  Rows are drawn in order from the global generator
    seeded with ``seed`` (restored afterwards), so the projection at a smaller
    width is the first rows of the projection at a larger width.
    """
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        dense = torch.zeros(width, feature_dim)
        for row in range(width):
            columns = torch.randperm(feature_dim)[:synaptic_degree]
            dense[row, columns] = torch.randn(synaptic_degree)
    return dense.to(device).to_sparse_csc()


def fly_codes(projection: torch.Tensor, features: torch.Tensor, coding_level: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices and values of the ``floor(rho m)`` largest entries of ``P x`` for every row of ``features``."""
    x = features.to(device=projection.device, dtype=projection.dtype)
    width = projection.shape[0]
    projected = torch.sparse.mm(projection, x.T)
    values, indices = projected.topk(max(1, int(width * coding_level)), dim=0, largest=True)
    return indices.T, values.T


def fly_code_cache(projection: torch.Tensor, features: torch.Tensor, coding_level: float,
                   batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """WTA codes of all rows, stored on the CPU as active indices and values."""
    width = projection.shape[0]
    active = max(1, int(width * coding_level))
    index_dtype = torch.int16 if width <= 32767 else torch.int32
    indices = torch.empty((len(features), active), dtype=index_dtype)
    values = torch.empty((len(features), active), dtype=torch.float32)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch_indices, batch_values = fly_codes(projection, features[start:stop], coding_level)
        indices[start:stop] = batch_indices.cpu().to(index_dtype)
        values[start:stop] = batch_values.cpu().to(torch.float32)
    return indices, values


def dense_codes(indices: torch.Tensor, values: torch.Tensor, width: int,
                device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dense WTA codes from cached active indices and values."""
    dense = torch.zeros((indices.shape[0], width), device=device, dtype=dtype)
    dense.scatter_(1, indices.to(device=device, dtype=torch.long), values.to(device=device, dtype=dtype))
    return dense


# ---------------------------------------------------------------------------
# RanPAC: Phi(x) = ReLU(x W) with a dense Gaussian matrix W
# ---------------------------------------------------------------------------
def ranpac_projection(feature_dim: int, width: int, seed: int, *, generated_width: int | None = None) -> torch.Tensor:
    """Gaussian matrix of shape ``(feature_dim, width)`` on the CPU.

    The matrix is drawn from a generator seeded with ``seed`` with
    ``generated_width`` columns (default ``width``) and its first ``width``
    columns are returned, so all widths of a width study share one draw.
    """
    generated_width = width if generated_width is None else generated_width
    if width > generated_width:
        raise ValueError("width exceeds the generated width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.randn(feature_dim, generated_width, generator=generator, dtype=torch.float32)
    return full if width == generated_width else full[:, :width].contiguous()


def ranpac_encode(features: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return torch.relu(features.to(device=projection.device, dtype=torch.float32) @ projection)


def encode_in_batches(encoder, features: torch.Tensor, indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.cat([encoder(features[indices[start : start + batch_size]]) for start in range(0, len(indices), batch_size)])


# ---------------------------------------------------------------------------
# CountSketch of the expanded features
# ---------------------------------------------------------------------------
class CountSketch:
    """Maps each of ``input_dim`` coordinates to one of ``sketch_dim`` outputs with a fixed random sign."""

    def __init__(self, input_dim: int, sketch_dim: int, seed: int, device: str | torch.device = "cpu") -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        buckets = torch.randint(sketch_dim, (input_dim,), generator=generator, dtype=torch.int32)
        signs = 2 * torch.randint(2, (input_dim,), generator=generator, dtype=torch.int8) - 1
        self.input_dim, self.sketch_dim = int(input_dim), int(sketch_dim)
        self.device = torch.device(device)
        self.buckets = buckets.to(device=self.device, dtype=torch.int32)
        self.signs = signs.to(device=self.device, dtype=torch.int8)

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(device=self.device, dtype=torch.float32)
        output = torch.zeros((len(values), self.sketch_dim), device=self.device, dtype=values.dtype)
        output.scatter_add_(1, self.buckets.to(torch.long).expand(len(values), -1), values * self.signs.to(values.dtype))
        return output

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {"countsketch.buckets": self.buckets, "countsketch.signs": self.signs}
