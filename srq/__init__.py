"""SRQ: square-root quantization of the ridge state of analytic continual learners."""

from .adaptive import AdaptiveFactor, batched_benefits, budget_bytes, greedy_select, score_blocks
from .codec import CompressedFactor, block_layout, factor_bytes
from .features import (
    CountSketch,
    dense_codes,
    encode_in_batches,
    fly_code_cache,
    fly_codes,
    fly_projection,
    ranpac_encode,
    ranpac_projection,
)
from .gram_int8 import Int8GramRidge, minimal_cholesky_load
from .qr import blocked_qr_update
from .ridge import ExactRidge, RidgeLearner, SquareRootRidge, cholesky_solve, sync, tensor_bytes, timed_update
from .selection_controls import SelectionControlRidge
from .state_accounting import learner_state_bytes, persistent_state_bytes, reported_exact_state_bytes

__all__ = [
    "AdaptiveFactor", "CompressedFactor", "CountSketch", "ExactRidge", "Int8GramRidge", "RidgeLearner",
    "SquareRootRidge", "batched_benefits", "block_layout", "blocked_qr_update", "budget_bytes",
    "cholesky_solve", "dense_codes", "encode_in_batches", "factor_bytes", "fly_code_cache", "fly_codes",
    "fly_projection", "greedy_select", "minimal_cholesky_load", "ranpac_encode", "ranpac_projection",
    "score_blocks", "sync", "tensor_bytes", "timed_update",
    "SelectionControlRidge", "learner_state_bytes", "persistent_state_bytes", "reported_exact_state_bytes",
]
