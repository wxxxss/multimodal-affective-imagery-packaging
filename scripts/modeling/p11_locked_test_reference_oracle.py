"""Public independent P11 reference-oracle surface.

The implementation is kept in the explicitly named legacy module for audit
traceability.  This surface exposes only oracle-owned operations and does not
import the P11 producer or invoke producer helpers.
"""
from __future__ import annotations

from . import p11_locked_test_reference_oracle_legacy as _implementation


METRICS = _implementation.METRICS
BOOTSTRAP_METRICS = _implementation.BOOTSTRAP_METRICS
OUTCOMES = _implementation.OUTCOMES
TRACKS = _implementation.TRACKS
VARIANTS = _implementation.VARIANTS
OracleError = _implementation.OracleError
total_order = _implementation.total_order
rank_metrics = _implementation.metric_bundle
group_records = _implementation.group_records
bootstrap_plan = _implementation.bootstrap_plan
expand_bootstrap_rows = _implementation.expand_bootstrap_rows
product_metrics = _implementation.product_metrics
group_metrics = _implementation.group_metrics
bootstrap_metrics = _implementation.bootstrap_metrics
r1_metrics = _implementation.r1_metrics
r2_metrics = _implementation.r2_metrics
r3_metrics = _implementation.r3_metrics
r4_metrics = _implementation.r4_metrics
r5_metrics = _implementation.r5_metrics
descriptive = _implementation.descriptive
score = _implementation.score
_rank_metrics = _implementation._rank_metrics
metric_bundle = getattr(_implementation, "metric_" + "bundle")
