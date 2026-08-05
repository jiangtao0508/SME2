#!/usr/bin/env python3
"""Compile and run the AArch64 SME FMOPA timing probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence


def run(argv: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", timeout=timeout, check=False,
    )


def compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        return shutil.which(configured) or configured
    for name in ("clang", "gcc", "cc"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("no C compiler found; set CC to the onsite compiler")


def parse_probe_output(text: str) -> Dict[str, Any]:
    streaming_vector_bytes: Optional[int] = None
    timer: Optional[str] = None
    records: Dict[str, List[Dict[str, int]]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split("\t")
        try:
            if len(fields) == 3 and fields[:2] == ["system", "streaming_vector_bytes"]:
                streaming_vector_bytes = int(fields[2])
            elif len(fields) == 3 and fields[:2] == ["system", "timer"]:
                timer = fields[2]
            elif len(fields) == 6 and fields[0] == "timing":
                records.setdefault(fields[1], []).append(
                    {
                        "operations_per_group": int(fields[2]),
                        "groups": int(fields[3]),
                        "ticks": int(fields[4]),
                        "timer_frequency_hz": int(fields[5]),
                    }
                )
            elif line.strip():
                raise ValueError("unknown record")
        except ValueError as error:
            raise ValueError(f"invalid SME probe output line {line_number}: {line!r}") from error
    required = {"baseline", "fmopa_one_tile", "fmopa_four_tiles"}
    if streaming_vector_bytes is None or not required.issubset(records):
        raise ValueError("SME probe output is incomplete")
    return {
        "streaming_vector_bytes": streaming_vector_bytes,
        "timer": timer,
        "records": records,
    }


def median_ticks_per_group(records: Sequence[Mapping[str, int]]) -> float:
    return statistics.median(record["ticks"] / record["groups"] for record in records)


def derive_profile(parsed: Mapping[str, Any]) -> Dict[str, Any]:
    records = parsed["records"]
    baseline = median_ticks_per_group(records["baseline"])
    one_total = median_ticks_per_group(records["fmopa_one_tile"])
    four_total = median_ticks_per_group(records["fmopa_four_tiles"])
    frequencies = [
        record["timer_frequency_hz"]
        for samples in records.values()
        for record in samples
    ]
    frequency = statistics.median(frequencies)
    one_ticks = one_total - baseline
    four_ticks = (four_total - baseline) / 4.0
    if one_ticks <= 0 or four_ticks <= 0:
        raise ValueError("FMOPA timing was not greater than the loop baseline")
    tick_ns = 1.0e9 / frequency
    streaming_vector_bytes = int(parsed["streaming_vector_bytes"])
    lanes_f32 = streaming_vector_bytes // 4
    if lanes_f32 <= 0:
        raise ValueError("invalid streaming vector length")
    flops_per_fmopa = 2 * lanes_f32 * lanes_f32
    one_ns = one_ticks * tick_ns
    throughput_ns = four_ticks * tick_ns
    return {
        "streaming_vector_bytes": streaming_vector_bytes,
        "f32_lanes_per_streaming_vector": lanes_f32,
        "f32_flops_per_fmopa": flops_per_fmopa,
        "architected_timer_frequency_hz": frequency,
        "loop_baseline_ticks_per_group": baseline,
        "one_tile_dependency_ticks_per_fmopa": one_ticks,
        "four_tile_throughput_ticks_per_fmopa": four_ticks,
        "one_tile_dependency_ns_per_fmopa": one_ns,
        "four_tile_throughput_ns_per_fmopa": throughput_ns,
        "four_tile_throughput_flops_per_ns": flops_per_fmopa / throughput_ns,
    }


def build_and_run(groups: int) -> Dict[str, Any]:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("SME timing must run natively on an AArch64 target")
    root = pathlib.Path(__file__).resolve().parent
    source_c = root / "sme_fmopa_probe.c"
    source_s = root / "sme_fmopa_probe.S"
    cc = compiler()
    with tempfile.TemporaryDirectory(prefix="sme-fmopa-timing-") as temporary:
        binary = pathlib.Path(temporary) / "sme_fmopa_probe"
        completed = run(
            [cc, "-O2", "-Wall", "-Wextra", "-march=armv9-a+sme", str(source_c), str(source_s), "-o", str(binary)]
        )
        if completed.returncode != 0:
            raise RuntimeError(f"SME timing probe compilation failed:\n{completed.stderr}")
        measured = run([str(binary), str(groups)], timeout=300)
        counter_sigill = measured.returncode == -4
        if counter_sigill:
            measured = run([str(binary), str(groups), "--clock"], timeout=300)
        if measured.returncode != 0:
            hint = ""
            if measured.returncode < 0:
                hint = " (the OS or CPU may not permit SME or CNTVCT_EL0)"
            raise RuntimeError(
                f"SME timing probe failed with exit {measured.returncode}{hint}:\n{measured.stderr}"
            )
    parsed = parse_probe_output(measured.stdout)
    derived = derive_profile(parsed)
    return {
        "schema_version": "1.0",
        "profile_id": "sme-fmopa-" + hashlib.sha256(
            (platform.machine() + platform.release() + str(derived["streaming_vector_bytes"])).encode()
        ).hexdigest()[:12],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "measured",
        "system": {
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "cpu": platform.processor() or None,
        },
        "measurement": parsed,
        "derived": derived,
        "quality": {
            "repetitions": len(parsed["records"]["baseline"]),
            "timer": parsed.get("timer") or "CNTVCT_EL0",
            "cntvct_sigill_fallback_used": counter_sigill,
            "includes_smstart_smstop_in_timed_region": False,
            "warnings": [
                "four-tile FMOPA throughput is a compute lower bound for a real K step; loads and loop scheduling are not included"
            ],
        },
        "provenance": {
            "compiler": cc,
            "c_sha256": hashlib.sha256(source_c.read_bytes()).hexdigest(),
            "assembly_sha256": hashlib.sha256(source_s.read_bytes()).hexdigest(),
        },
    }


def summary(profile: Mapping[str, Any]) -> str:
    derived = profile["derived"]
    return "\n".join(
        [
            "SME FMOPA Timing Summary",
            f"profile_id={profile['profile_id']}",
            f"streaming_vector_bytes={derived['streaming_vector_bytes']}",
            f"one_tile_dependency_ns_per_fmopa={derived['one_tile_dependency_ns_per_fmopa']}",
            f"four_tile_throughput_ns_per_fmopa={derived['four_tile_throughput_ns_per_fmopa']}",
            f"four_tile_throughput_flops_per_ns={derived['four_tile_throughput_flops_per_ns']}",
        ]
    ) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--groups", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if args.groups <= 0:
        parser.error("--groups must be positive")
    try:
        profile = build_and_run(args.groups)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "SmeTimingProfile.v1.json"
        path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"calibrate_sme.py: {error}", file=sys.stderr)
        return 1
    print(summary(profile), end="")
    print(f"SME_TIMING_PROFILE={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
