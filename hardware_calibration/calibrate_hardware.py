#!/usr/bin/env python3
"""Build and run numeric microbenchmarks for HardwareProfile 1.1."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SIZE_RE = re.compile(r"^([0-9]+)([KMG]?)$", re.IGNORECASE)


def parse_size(text: str) -> Optional[int]:
    match = SIZE_RE.fullmatch(text.strip())
    if not match:
        return None
    value = int(match.group(1))
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return value * multiplier


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_number(path: Path, scale: float = 1.0) -> Optional[float]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return float(text) * scale
    except ValueError:
        return None


def run(argv: Sequence[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def discover_linux_cache(root: Path = Path("/sys/devices/system/cpu/cpu0/cache")) -> List[Dict[str, object]]:
    caches: List[Dict[str, object]] = []
    if not root.is_dir():
        return caches
    for index in sorted(root.glob("index*")):
        level_text = read_text(index / "level")
        kind = read_text(index / "type")
        size_text = read_text(index / "size")
        if level_text is None or kind is None or size_text is None:
            continue
        size_bytes = parse_size(size_text)
        try:
            level = int(level_text)
        except ValueError:
            continue
        caches.append(
            {
                "level": level,
                "type": kind,
                "size_bytes": size_bytes,
                "line_bytes": parse_size(read_text(index / "coherency_line_size") or ""),
                "ways": parse_size(read_text(index / "ways_of_associativity") or ""),
                "sets": parse_size(read_text(index / "number_of_sets") or ""),
                "shared_cpu_list": read_text(index / "shared_cpu_list"),
            }
        )
    return caches


def sysctl_number(name: str) -> Optional[int]:
    executable = shutil.which("sysctl")
    if executable is None:
        return None
    completed = run([executable, "-n", name], timeout=10)
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def discover_macos_cache() -> List[Dict[str, object]]:
    if platform.system() != "Darwin":
        return []
    line = sysctl_number("hw.cachelinesize")
    result = []
    for level, name in ((1, "hw.l1dcachesize"), (2, "hw.l2cachesize"), (3, "hw.l3cachesize")):
        size = sysctl_number(name)
        if size:
            result.append(
                {
                    "level": level,
                    "type": "Data" if level == 1 else "Unified",
                    "size_bytes": size,
                    "line_bytes": line,
                    "ways": None,
                    "sets": None,
                    "shared_cpu_list": None,
                }
            )
    return result


def discover_cpu_model() -> Optional[str]:
    if platform.system() == "Linux":
        text = read_text(Path("/proc/cpuinfo")) or ""
        for label in ("model name", "CPU part", "Processor", "Hardware"):
            match = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
    if platform.system() == "Darwin":
        executable = shutil.which("sysctl")
        if executable:
            completed = run([executable, "-n", "machdep.cpu.brand_string"], timeout=10)
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
    return platform.processor() or None


def discover_frequency_ghz() -> Tuple[Optional[float], Optional[str]]:
    candidates = (
        (Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"), 1.0e-6, "sysfs_scaling_cur_freq"),
        (Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"), 1.0e-6, "sysfs_cpuinfo_max_freq"),
    )
    for path, scale, source in candidates:
        value = read_number(path, scale)
        if value and value > 0:
            return value, source
    if platform.system() == "Darwin":
        value = sysctl_number("hw.cpufrequency")
        if value:
            return value / 1.0e9, "sysctl_hw_cpufrequency"
    return None, None


def select_cache(caches: Sequence[Mapping[str, object]], level: int) -> Optional[int]:
    eligible = [
        int(cache["size_bytes"])
        for cache in caches
        if cache.get("level") == level
        and cache.get("type") in {"Data", "Unified"}
        and isinstance(cache.get("size_bytes"), int)
    ]
    return max(eligible) if eligible else None


def cache_line_bytes(caches: Sequence[Mapping[str, object]]) -> Optional[int]:
    lines = [int(cache["line_bytes"]) for cache in caches if isinstance(cache.get("line_bytes"), int)]
    return min(lines) if lines else None


def parse_probe_output(text: str) -> Dict[str, object]:
    latency: Dict[int, float] = {}
    stride: Dict[int, float] = {}
    prefetch: Dict[int, float] = {}
    bandwidth: Optional[Dict[str, object]] = None
    reported_system: Dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split("\t")
        try:
            if len(fields) == 3 and fields[0] == "latency":
                latency[int(fields[1])] = float(fields[2])
            elif len(fields) == 4 and fields[0] == "bandwidth":
                bandwidth = {
                    "working_set_bytes": int(fields[1]),
                    "rounds": int(fields[2]),
                    "bytes_per_ns": float(fields[3]),
                }
            elif len(fields) == 3 and fields[0] == "stride":
                stride[int(fields[1])] = float(fields[2])
            elif len(fields) == 3 and fields[0] == "prefetch":
                prefetch[int(fields[1])] = float(fields[2])
            elif len(fields) == 3 and fields[0] == "system" and fields[1] in {
                "cache_line_bytes", "l1_cache_bytes", "l2_cache_bytes", "l3_cache_bytes"
            }:
                reported_system[fields[1]] = int(fields[2])
            elif line.strip():
                raise ValueError("unknown record")
        except ValueError as error:
            raise ValueError(f"invalid probe output line {line_number}: {line!r}") from error
    if not latency or bandwidth is None or not stride or 0 not in prefetch:
        raise ValueError("probe output is incomplete")
    return {
        "latency": latency,
        "bandwidth": bandwidth,
        "stride": stride,
        "prefetch": prefetch,
        "reported_system": reported_system,
    }


def aggregate_probe_runs(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not runs:
        raise ValueError("at least one probe run is required")

    def aggregate_map(section: str) -> List[Dict[str, object]]:
        keys = sorted(set.intersection(*(set(run[section]) for run in runs)))  # type: ignore[arg-type]
        output = []
        for key in keys:
            samples = [float(run[section][key]) for run in runs]  # type: ignore[index]
            output.append(
                {
                    "key": key,
                    "median": statistics.median(samples),
                    "minimum": min(samples),
                    "maximum": max(samples),
                    "samples": samples,
                }
            )
        return output

    bandwidth_samples = [float(run["bandwidth"]["bytes_per_ns"]) for run in runs]  # type: ignore[index]
    first_bandwidth = runs[0]["bandwidth"]  # type: ignore[index]
    reported_system = {}
    for key in ("cache_line_bytes", "l1_cache_bytes", "l2_cache_bytes", "l3_cache_bytes"):
        samples = [
            int(run["reported_system"][key])
            for run in runs
            if isinstance(run.get("reported_system"), Mapping)
            and isinstance(run["reported_system"].get(key), int)
        ]
        reported_system[key] = int(statistics.median(samples)) if samples else None
    return {
        "latency": aggregate_map("latency"),
        "stride": aggregate_map("stride"),
        "prefetch": aggregate_map("prefetch"),
        "bandwidth": {
            "working_set_bytes": int(first_bandwidth["working_set_bytes"]),
            "rounds": int(first_bandwidth["rounds"]),
            "bytes_per_ns_median": statistics.median(bandwidth_samples),
            "bytes_per_ns_minimum": min(bandwidth_samples),
            "bytes_per_ns_maximum": max(bandwidth_samples),
            "samples": bandwidth_samples,
        },
        "reported_system": reported_system,
    }


def point_median(points: Sequence[Mapping[str, object]], key: int) -> Optional[float]:
    for point in points:
        if point["key"] == key:
            return float(point["median"])
    return None


def representative_latency(
    points: Sequence[Mapping[str, object]], lower_exclusive: int, upper_inclusive: int
) -> Optional[float]:
    values = [
        float(point["median"])
        for point in points
        if lower_exclusive < int(point["key"]) <= upper_inclusive
    ]
    return statistics.median(values) if values else None


def derive_profile(
    aggregated: Mapping[str, object],
    caches: Sequence[Mapping[str, object]],
    frequency_ghz: Optional[float],
) -> Tuple[Dict[str, object], List[str]]:
    warnings: List[str] = []
    latency_points = aggregated["latency"]  # type: ignore[assignment]
    stride_points = aggregated["stride"]  # type: ignore[assignment]
    prefetch_points = aggregated["prefetch"]  # type: ignore[assignment]
    reported_system = aggregated.get("reported_system", {})
    if not isinstance(reported_system, Mapping):
        reported_system = {}
    l1_bytes = select_cache(caches, 1) or reported_system.get("l1_cache_bytes")
    l2_bytes = select_cache(caches, 2) or reported_system.get("l2_cache_bytes")
    largest_cache = max(
        (int(cache["size_bytes"]) for cache in caches if isinstance(cache.get("size_bytes"), int)),
        default=0,
    )
    largest_latency_point = max(latency_points, key=lambda point: int(point["key"]))
    largest_working_set = int(largest_latency_point["key"])
    memory_latency_ns = float(largest_latency_point["median"])
    if largest_cache and largest_working_set <= 2 * largest_cache:
        warnings.append(
            "largest pointer-chase working set is not more than 2x the largest reported cache; memory latency may be underestimated"
        )

    l1_latency_ns = (
        representative_latency(latency_points, 0, max(4096, (l1_bytes or 32768) // 2))
    )
    l2_latency_ns = None
    if l1_bytes and l2_bytes:
        l2_latency_ns = representative_latency(latency_points, l1_bytes * 2, max(l1_bytes * 2, l2_bytes // 2))

    stride64 = point_median(stride_points, 64)
    effective_stride = None
    if stride64 and stride64 > 0:
        eligible = [
            int(point["key"])
            for point in stride_points
            if float(point["median"]) <= 1.25 * stride64
        ]
        effective_stride = max(eligible) if eligible else None

    baseline_prefetch_loop = point_median(prefetch_points, 0)
    issue_costs = []
    for point in prefetch_points:
        issue_every = int(point["key"])
        if issue_every == 0 or baseline_prefetch_loop is None:
            continue
        delta_per_iteration = float(point["median"]) - baseline_prefetch_loop
        issue_costs.append(
            {
                "issue_every": issue_every,
                "loop_ns_per_iteration": float(point["median"]),
                "delta_ns_per_iteration": delta_per_iteration,
                "estimated_ns_per_issued_prefetch": delta_per_iteration * issue_every,
            }
        )
    positive_issue_costs = [
        item["estimated_ns_per_issued_prefetch"]
        for item in issue_costs
        if item["estimated_ns_per_issued_prefetch"] > 0
    ]
    prefetch_cost_ns = statistics.median(positive_issue_costs) if positive_issue_costs else None
    if prefetch_cost_ns is None:
        warnings.append("hot-buffer prefetch loop did not expose a positive standalone PRFM cost")

    bandwidth = aggregated["bandwidth"]  # type: ignore[assignment]
    memory_latency_cycles = memory_latency_ns * frequency_ghz if frequency_ghz else None
    prefetch_cost_cycles = prefetch_cost_ns * frequency_ghz if frequency_ghz and prefetch_cost_ns else None

    return {
        "cache_line_bytes": cache_line_bytes(caches) or reported_system.get("cache_line_bytes"),
        "l1d_bytes": l1_bytes,
        "l2_bytes": l2_bytes,
        "l1_hit_latency_ns": l1_latency_ns,
        "l2_hit_latency_ns": l2_latency_ns,
        "memory_latency_ns": memory_latency_ns,
        "memory_latency_cycles_estimate": memory_latency_cycles,
        "sustainable_bandwidth_bytes_per_ns": float(bandwidth["bytes_per_ns_median"]),
        "prefetch_instruction_cost_ns": prefetch_cost_ns,
        "prefetch_instruction_cost_cycles_estimate": prefetch_cost_cycles,
        "prefetch_issue_measurements": issue_costs,
        "hardware_prefetch_effective_stride_bytes_heuristic": effective_stride,
        "max_outstanding_prefetches": None,
    }, warnings


def compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        resolved = shutil.which(configured) or configured
        if Path(resolved).exists():
            return resolved
    for name in ("cc", "clang", "gcc"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise RuntimeError("no C compiler found; set CC to the onsite compiler")


def compile_probe(source: Path, binary: Path, assembly: Path, cc: str) -> Dict[str, object]:
    common = ["-O3", "-std=c11", "-Wall", "-Wextra"]
    compile_result = run([cc, *common, str(source), "-o", str(binary)])
    if compile_result.returncode != 0:
        raise RuntimeError(f"hardware probe compilation failed:\n{compile_result.stderr}")
    assembly_result = run([cc, *common, "-S", str(source), "-o", str(assembly)])
    assembly_text = read_text(assembly) or ""
    prefetch_pattern = re.compile(r"\b(?:prfm|prefetch[a-z]*)\b", re.IGNORECASE)
    return {
        "compiler": cc,
        "compile_stderr": compile_result.stderr,
        "assembly_prefetch_instruction_found": bool(prefetch_pattern.search(assembly_text)),
        "assembly_sha256": hashlib.sha256(assembly_text.encode("utf-8")).hexdigest(),
        "assembly_result": assembly_result.returncode,
    }


def build_profile(quick: bool, repetitions: int) -> Dict[str, object]:
    source = Path(__file__).with_name("hardware_probe.c")
    caches = discover_linux_cache() or discover_macos_cache()
    frequency_ghz, frequency_source = discover_frequency_ghz()
    cc = compiler()
    with tempfile.TemporaryDirectory(prefix="sme-hardware-calibration-") as temporary:
        temporary_path = Path(temporary)
        binary = temporary_path / "hardware_probe"
        assembly = temporary_path / "hardware_probe.s"
        build = compile_probe(source, binary, assembly, cc)
        parsed_runs = []
        raw_outputs = []
        for _ in range(repetitions):
            argv = [str(binary)] + (["--quick"] if quick else [])
            completed = run(argv, timeout=300)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"hardware probe failed with exit {completed.returncode}:\n{completed.stderr}"
                )
            parsed_runs.append(parse_probe_output(completed.stdout))
            raw_outputs.append(completed.stdout)

    aggregated = aggregate_probe_runs(parsed_runs)
    derived, warnings = derive_profile(aggregated, caches, frequency_ghz)
    if not build["assembly_prefetch_instruction_found"]:
        warnings.append("compiler assembly did not contain a recognizable PRFM/prefetch instruction")
    if frequency_ghz is None:
        warnings.append("CPU frequency unavailable; nanosecond values are authoritative and cycle estimates are null")

    identity = {
        "architecture": platform.machine(),
        "cpu_model": discover_cpu_model(),
        "cache": caches,
    }
    profile_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {
        "schema_version": "1.1",
        "profile_id": f"measured-{platform.machine()}-{profile_digest}",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "measured",
        "system": {
            "os": platform.system(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "cpu_model": discover_cpu_model(),
            "logical_cpus": os.cpu_count(),
            "frequency_ghz_observed": frequency_ghz,
            "frequency_source": frequency_source,
        },
        "cache_topology": caches,
        "measurements": aggregated,
        "derived": derived,
        "quality": {
            "mode": "quick" if quick else "full",
            "repetitions": repetitions,
            "warnings": warnings,
            "prefetch_instruction_verified_in_assembly": build["assembly_prefetch_instruction_found"],
        },
        "provenance": {
            "tool": "SME2 hardware_calibration",
            "tool_version": "0.1.0",
            "compiler": build["compiler"],
            "probe_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "assembly_sha256": build["assembly_sha256"],
        },
    }


def summary_text(profile: Mapping[str, object]) -> str:
    derived = profile["derived"]  # type: ignore[assignment]
    quality = profile["quality"]  # type: ignore[assignment]
    lines = [
        "SME Hardware Calibration Summary",
        f"profile_id={profile['profile_id']}",
        f"mode={quality['mode']} repetitions={quality['repetitions']}",
        f"cache_line_bytes={derived['cache_line_bytes']}",
        f"l1d_bytes={derived['l1d_bytes']}",
        f"l2_bytes={derived['l2_bytes']}",
        f"l1_hit_latency_ns={derived['l1_hit_latency_ns']}",
        f"l2_hit_latency_ns={derived['l2_hit_latency_ns']}",
        f"memory_latency_ns={derived['memory_latency_ns']}",
        f"sustainable_bandwidth_bytes_per_ns={derived['sustainable_bandwidth_bytes_per_ns']}",
        f"prefetch_instruction_cost_ns={derived['prefetch_instruction_cost_ns']}",
        f"hardware_prefetch_effective_stride_bytes_heuristic={derived['hardware_prefetch_effective_stride_bytes_heuristic']}",
        f"prefetch_instruction_verified={quality['prefetch_instruction_verified_in_assembly']}",
    ]
    for warning in quality["warnings"]:
        lines.append(f"WARNING={warning}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    repetitions = args.repetitions if args.repetitions is not None else (1 if args.quick else 3)
    if repetitions <= 0:
        parser.error("--repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        profile = build_profile(args.quick, repetitions)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        print(f"calibrate_hardware.py: {error}", file=sys.stderr)
        return 1
    profile_path = args.output / "HardwareProfile.v1.1.json"
    summary_path = args.output / "HARDWARE_CALIBRATION_SUMMARY.txt"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary_text(profile), encoding="utf-8")
    print(summary_text(profile), end="")
    print(f"HARDWARE_PROFILE={profile_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
