#!/usr/bin/env python3
"""Derive one packed-GEMM RHS prefetch plan from measured numeric inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path} must be a positive number")
    return float(value)


def optional_positive(value: Any, path: str) -> Optional[float]:
    if value is None:
        return None
    return positive_number(value, path)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def select_candidate(kernel: Mapping[str, Any], candidate_id: int) -> Mapping[str, Any]:
    if kernel.get("schema_version") != "1.0":
        raise ValueError("GemmKernelProfile schema_version must be 1.0")
    candidates = kernel.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("GemmKernelProfile.candidates must be an array")
    matches = [item for item in candidates if isinstance(item, dict) and item.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected one GEMM candidate with candidate_id={candidate_id}")
    return matches[0]


def measured_hardware(hardware: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if hardware.get("schema_version") != "1.1" or hardware.get("source") != "measured":
        raise ValueError("HardwareProfile must be a measured schema_version 1.1 profile")
    derived = hardware.get("derived")
    if not isinstance(derived, dict):
        raise ValueError("HardwareProfile.derived must be an object")
    quality = hardware.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("HardwareProfile.quality must be an object")
    if not quality.get("prefetch_instruction_verified_in_assembly"):
        raise ValueError("HardwareProfile did not verify a compiler prefetch instruction")
    return derived, quality


def derive_plan(
    hardware: Mapping[str, Any],
    kernel: Mapping[str, Any],
    anchor_step_ns: float,
    candidate_id: int = 0,
    overhead_fraction: float = 0.10,
    bandwidth_fraction: float = 0.25,
) -> Dict[str, Any]:
    anchor_step_ns = positive_number(anchor_step_ns, "anchor_step_ns")
    if not 0 < overhead_fraction <= 1:
        raise ValueError("overhead_fraction must be in (0, 1]")
    if not 0 < bandwidth_fraction <= 1:
        raise ValueError("bandwidth_fraction must be in (0, 1]")
    derived, quality = measured_hardware(hardware)
    feature = select_candidate(kernel, candidate_id)

    cache_line = int(positive_number(derived.get("cache_line_bytes"), "derived.cache_line_bytes"))
    memory_latency_ns = positive_number(derived.get("memory_latency_ns"), "derived.memory_latency_ns")
    bandwidth = positive_number(
        derived.get("sustainable_bandwidth_bytes_per_ns"),
        "derived.sustainable_bandwidth_bytes_per_ns",
    )
    l1_bytes = int(positive_number(derived.get("l1d_bytes"), "derived.l1d_bytes"))
    l2_bytes = int(positive_number(derived.get("l2_bytes"), "derived.l2_bytes"))
    prefetch_cost_ns = optional_positive(
        derived.get("prefetch_instruction_cost_ns"), "derived.prefetch_instruction_cost_ns"
    )
    max_outstanding = optional_positive(
        derived.get("max_outstanding_prefetches"), "derived.max_outstanding_prefetches"
    )

    vector_read_bytes = int(positive_number(feature.get("vector_read_bytes"), "vector_read_bytes"))
    source_allocation_bytes = int(
        positive_number(feature.get("source_allocation_bytes"), "source_allocation_bytes")
    )
    rhs_row_bytes = int(positive_number(feature.get("rhs_row_bytes"), "rhs_row_bytes"))
    trip_count = int(positive_number(feature.get("loop_trip_count"), "loop_trip_count"))
    coverage_lines = max(1, math.ceil(vector_read_bytes / cache_line))
    coverage_bytes = coverage_lines * cache_line

    # Every K iteration advances to a distinct RHS row. Start at one issue per
    # iteration, then throttle only when measured instruction or bandwidth cost
    # exceeds an explicit per-step budget.
    issue_every = 1
    if prefetch_cost_ns is not None:
        issue_every = max(
            issue_every,
            math.ceil((coverage_lines * prefetch_cost_ns) / (anchor_step_ns * overhead_fraction)),
        )
    issue_every = max(
        issue_every,
        math.ceil(coverage_bytes / (anchor_step_ns * bandwidth * bandwidth_fraction)),
    )

    required_distance = max(1, math.ceil(memory_latency_ns / anchor_step_ns))
    outstanding_lines = math.ceil(required_distance / issue_every) * coverage_lines
    if max_outstanding is not None and outstanding_lines > max_outstanding:
        issue_every = max(issue_every, math.ceil(required_distance * coverage_lines / max_outstanding))
        outstanding_lines = math.ceil(required_distance / issue_every) * coverage_lines

    warnings = list(quality.get("warnings", [])) if isinstance(quality.get("warnings"), list) else []
    rejection: Optional[str] = None
    if required_distance >= trip_count:
        rejection = "REQUIRED_DISTANCE_EXCEEDS_K_LOOP"
    elif issue_every >= trip_count:
        rejection = "ISSUE_THROTTLE_EXCEEDS_K_LOOP"

    l1_occupancy = outstanding_lines * cache_line
    # The packed source remains live across the reduction. L1 is selected only
    # when the observed RHS allocation plus in-flight lines consumes no more
    # than half of L1, leaving space for the LHS and accumulator traffic.
    target_cache = (
        "L1"
        if source_allocation_bytes + l1_occupancy <= int(0.50 * l1_bytes)
        else "L2"
    )
    target_capacity = l1_bytes if target_cache == "L1" else l2_bytes
    if source_allocation_bytes + l1_occupancy > int(0.80 * target_capacity):
        warnings.append("packed RHS plus prefetch occupancy exceeds 80% of selected target cache")
    if prefetch_cost_ns is None:
        warnings.append("prefetch instruction cost unavailable; issue_every was not cost-throttled")
    if max_outstanding is None:
        warnings.append("max outstanding prefetches unavailable; outstanding limit was not applied")

    decisions = []
    if rejection is None:
        decisions.append(
            {
                "decision_id": f"gemm-rhs-{candidate_id}",
                "object_id": "packed_rhs_panel",
                "strategy": "TILE_PREFETCH",
                "distance": {"value": required_distance, "unit": "ITERATION", "anchor_loop": "k_loop"},
                "target_cache": target_cache,
                "granularity": {"kind": "PANEL", "bytes": coverage_bytes},
                "emission": {"issue_every": issue_every, "coverage_lines": coverage_lines},
                "model": {
                    "anchor_step_ns": anchor_step_ns,
                    "memory_latency_ns": memory_latency_ns,
                    "rhs_row_bytes": rhs_row_bytes,
                    "source_allocation_bytes": source_allocation_bytes,
                    "vector_read_bytes": vector_read_bytes,
                    "estimated_outstanding_lines": outstanding_lines,
                    "estimated_prefetch_bytes_per_step": coverage_bytes / issue_every,
                },
                "reasons": [
                    "distance=ceil(memory_latency_ns/anchor_step_ns)",
                    "coverage covers the observed RHS vector read",
                    "issue frequency is bounded by measured issue and bandwidth budgets",
                ],
            }
        )

    identity = {
        "hardware_profile_id": hardware.get("profile_id"),
        "candidate_id": candidate_id,
        "anchor_step_ns": anchor_step_ns,
        "overhead_fraction": overhead_fraction,
        "bandwidth_fraction": bandwidth_fraction,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "schema_version": "1.1",
        "plan_id": f"gemm-rhs-plan-{digest}",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware_profile": {
            "profile_id": hardware.get("profile_id"),
            "cache_line_bytes": cache_line,
            "source": "measured",
        },
        "kernel_profile": {"schema_version": kernel.get("schema_version"), "candidate_id": candidate_id},
        "model_config": {
            "overhead_fraction": overhead_fraction,
            "bandwidth_fraction": bandwidth_fraction,
            "no_parameter_search": True,
        },
        "decisions": decisions,
        "diagnostics": {"rejection": rejection, "warnings": warnings},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", type=pathlib.Path, required=True)
    parser.add_argument("--kernel", type=pathlib.Path, required=True)
    parser.add_argument("--anchor-step-ns", type=float, required=True)
    parser.add_argument("--candidate-id", type=int, default=0)
    parser.add_argument("--overhead-fraction", type=float, default=0.10)
    parser.add_argument("--bandwidth-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = derive_plan(
            load_json(args.hardware),
            load_json(args.kernel),
            args.anchor_step_ns,
            args.candidate_id,
            args.overhead_fraction,
            args.bandwidth_fraction,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"plan_gemm_rhs.py: {error}", file=sys.stderr)
        return 1
    print(f"PREFETCH_PLAN={args.output.resolve()}")
    if not plan["decisions"]:
        print(f"NO_PREFETCH={plan['diagnostics']['rejection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
