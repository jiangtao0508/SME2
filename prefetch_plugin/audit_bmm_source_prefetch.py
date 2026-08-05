#!/usr/bin/env python3
"""Fail closed unless a rewritten BMM payload prefetches the original source."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--source-argument", type=int, required=True)
    parser.add_argument("--expected-prefetches", type=int, required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--issue-every", type=int, required=True)
    parser.add_argument("--locality", type=int, required=True)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    text = args.input.read_text()
    prefetch_lines = [
        line for line in text.splitlines() if "memref.prefetch" in line
    ]
    source_cast_matches = list(
        re.finditer(
            rf"^\s*(%[-\w.$]+)\s*=\s*memref\.reinterpret_cast\s+"
            rf"%arg{args.source_argument}\b.*"
            rf"prefetch\.distance_iterations = {args.distance} : i64.*"
            rf"prefetch\.source_argument = {args.source_argument} : i64",
            text,
            flags=re.MULTILINE,
        )
    )
    future_view = (
        source_cast_matches[0].group(1) if len(source_cast_matches) == 1 else None
    )

    checks = {
        "prefetch_count": len(prefetch_lines) == args.expected_prefetches,
        "all_prefetches_use_future_source_view": future_view is not None
        and all(f"memref.prefetch {future_view}[" in line for line in prefetch_lines),
        "exactly_one_tagged_future_source_view": len(source_cast_matches) == 1,
        "issue_frequency_tag_present": (
            f"prefetch.issue_every = {args.issue_every} : i64" in text
        ),
        "locality_matches": all(
            f"locality<{args.locality}>" in line for line in prefetch_lines
        ),
        "no_private_alloc_prefetch_target": all(
            "%alloc" not in line for line in prefetch_lines
        ),
        "prefetch_precedes_first_copy": (
            text.find("memref.prefetch") >= 0
            and text.find("memref.copy") >= 0
            and text.find("memref.prefetch") < text.find("memref.copy")
        ),
    }
    report = {
        "schema": "bmm-source-prefetch-audit-v1",
        "input": str(args.input.resolve()),
        "source_argument": args.source_argument,
        "distance_iterations": args.distance,
        "issue_every": args.issue_every,
        "locality": args.locality,
        "expected_prefetches": args.expected_prefetches,
        "observed_prefetches": len(prefetch_lines),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered)
    print(rendered, end="")
    if report["status"] != "passed":
        print("source-A prefetch structural audit failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
