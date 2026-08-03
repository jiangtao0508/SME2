#!/usr/bin/env python3
"""Resolve one PrefetchPlan 1.0 GEMM decision to plugin options."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Tuple


LOCALITY_BY_CACHE = {"L1": 3, "L2": 2}


def resolve(
    plan: dict, decision_id: Optional[str] = None
) -> Tuple[int, int, int, int, str, str]:
    if plan.get("schema_version") != "1.0":
        raise ValueError("expected PrefetchPlan schema_version 1.0")
    decisions = plan.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("plan has no enabled prefetch decisions")

    if decision_id is None:
        if len(decisions) != 1:
            raise ValueError("plan has multiple decisions; select one with --decision-id")
        decision = decisions[0]
    else:
        matches = [item for item in decisions if item.get("decision_id") == decision_id]
        if len(matches) != 1:
            raise ValueError(f"expected one decision named {decision_id!r}")
        decision = matches[0]

    if decision.get("strategy") != "TILE_PREFETCH":
        raise ValueError("prefetch-gemm-rhs currently requires TILE_PREFETCH")
    distance = decision.get("distance")
    if not isinstance(distance, dict) or distance.get("unit") != "ITERATION":
        raise ValueError("prefetch-gemm-rhs distance must use ITERATION units")
    distance_value = distance.get("value")
    if isinstance(distance_value, bool) or not isinstance(distance_value, int) or distance_value <= 0:
        raise ValueError("distance.value must be a positive integer")
    target_cache = decision.get("target_cache")
    if target_cache not in LOCALITY_BY_CACHE:
        raise ValueError("target_cache must be L1 or L2")
    granularity_spec = decision.get("granularity", {})
    granularity = granularity_spec.get("kind")
    if granularity not in {"PANEL", "TILE"}:
        raise ValueError("prefetch-gemm-rhs requires PANEL or TILE granularity")

    decision_name = decision.get("decision_id")
    object_id = decision.get("object_id")
    if not isinstance(decision_name, str) or not decision_name:
        raise ValueError("decision_id must be a non-empty string")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("object_id must be a non-empty string")
    if any("\t" in value or "\n" in value for value in (decision_name, object_id)):
        raise ValueError("decision_id and object_id cannot contain tabs or newlines")

    cache_line_bytes = plan.get("hardware_profile", {}).get("cache_line_bytes")
    if (
        isinstance(cache_line_bytes, bool)
        or not isinstance(cache_line_bytes, int)
        or cache_line_bytes <= 0
    ):
        raise ValueError("hardware_profile.cache_line_bytes must be a positive integer")
    coverage_bytes = granularity_spec.get("bytes")
    if coverage_bytes is None:
        coverage_lines = 1
    elif isinstance(coverage_bytes, bool) or not isinstance(coverage_bytes, int) or coverage_bytes <= 0:
        raise ValueError("granularity.bytes must be a positive integer")
    else:
        coverage_lines = (coverage_bytes + cache_line_bytes - 1) // cache_line_bytes

    return (
        distance_value,
        LOCALITY_BY_CACHE[target_cache],
        coverage_lines,
        cache_line_bytes,
        decision_name,
        object_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=pathlib.Path)
    parser.add_argument("--decision-id")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        values = resolve(plan, args.decision_id)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"prefetch_plan_options.py: {error}", file=sys.stderr)
        return 1
    print("\t".join(str(value) for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
