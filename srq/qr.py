"""Blocked Householder QR update of an upper-triangular factor."""

from __future__ import annotations

import torch


def blocked_qr_update(
    upper: torch.Tensor,
    rows: torch.Tensor,
    *,
    panel_size: int = 128,
    trailing_chunk_size: int | None = None,
    preserve_rows: bool = True,
) -> torch.Tensor:
    """Return the positive-diagonal triangular factor of ``[upper; rows]``.

    If ``R`` is upper triangular and ``Phi`` has the same number of columns,
    the result ``R_new`` satisfies ``R_new.T @ R_new = R.T @ R + Phi.T @ Phi``.
    Columns are eliminated one panel of ``panel_size`` columns at a time with
    compact Householder reflectors, which touch only the panel rows of
    ``upper`` and the rows of ``rows``.  ``upper`` is overwritten and returned.
    ``rows`` is cloned unless ``preserve_rows`` is False.  The panel size is a
    unit of computation and does not change the factor beyond floating-point
    rounding.
    """
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1] or not len(upper):
        raise ValueError("upper must be a non-empty square matrix")
    if rows.ndim != 2 or rows.shape[1] != len(upper) or not len(rows):
        raise ValueError("rows must be a non-empty matrix with the factor's columns")
    if panel_size <= 0 or (trailing_chunk_size is not None and trailing_chunk_size <= 0):
        raise ValueError("panel and chunk sizes must be positive")
    if upper.device != rows.device or upper.dtype != rows.dtype:
        raise ValueError("upper and rows must share device and dtype")

    dimension = len(upper)
    residual = rows.clone() if preserve_rows else rows
    for start in range(0, dimension, panel_size):
        end = min(start + panel_size, dimension)
        width = end - start
        panel = torch.cat((upper[start:end, start:end], residual[:, start:end]), dim=0)
        reflectors, tau = torch.geqrf(panel)
        diagonal_block = torch.triu(reflectors[:width])
        # QR is unique only up to row signs; a positive diagonal is the Cholesky factor.
        signs = torch.where(
            diagonal_block.diagonal() < 0,
            -torch.ones((), device=upper.device, dtype=upper.dtype),
            torch.ones((), device=upper.device, dtype=upper.dtype),
        )
        trailing_width = dimension - end if trailing_chunk_size is None else trailing_chunk_size
        if end < dimension:
            for column_start in range(end, dimension, trailing_width):
                column_end = min(column_start + trailing_width, dimension)
                trailing = torch.cat(
                    (upper[start:end, column_start:column_end], residual[:, column_start:column_end]), dim=0
                )
                transformed = torch.ormqr(reflectors, tau, trailing, left=True, transpose=True)
                upper[start:end, column_start:column_end].copy_(signs[:, None] * transformed[:width])
                residual[:, column_start:column_end].copy_(transformed[width:])
        upper[start:end, start:end].copy_(signs[:, None] * diagonal_block)
    return upper
