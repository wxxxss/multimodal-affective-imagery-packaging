#!/usr/bin/env python3
"""Canonical annotation schema-level rationale policy for v2.1 validation.

This module is the SINGLE home of the canonical rationale policy.  It is
imported by the validator, the decision-sidecar helper, and the normalizer.
The mapping contract never controls rationale requirements, and no second
implementation exists.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .prepare_affective_imagery_validation_v21 import _clean_text
except ImportError:  # pragma: no cover - direct script execution
    from prepare_affective_imagery_validation_v21 import _clean_text


def rationale_required(row: pd.Series, item_type: str) -> bool:
    """Return whether canonical v2.1 schema policy requires a rationale."""
    if _clean_text(row.get("human_confidence")) == "low":
        return True
    if item_type == "sentence":
        if _clean_text(row.get("human_action")) != "keep":
            return True
        for column in [
            "human_packaging_visual",
            "human_relation_valid",
            "human_package_level",
            "human_dimension_code",
            "human_polarity",
        ]:
            if _clean_text(row.get(column)) == "uncertain":
                return True
        return False

    model_label_value = int(float(row.get("model_label_value", 0) or 0))
    if model_label_value == 1:
        return _clean_text(row.get("human_product_label_traceable")) != "yes"
    return _clean_text(row.get("human_unlabeled_missed_signal")) != "no"
