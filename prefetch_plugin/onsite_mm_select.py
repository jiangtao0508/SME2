#!/usr/bin/env python3
"""Select or force one existing Kunpeng FlagGems MM autotune config."""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys


def config_record(config, timings=None):
    record = {
        "meta": dict(config.kwargs),
        "num_stages": config.num_stages,
        "num_warps": config.num_warps,
    }
    if timings is not None:
        if isinstance(timings, (list, tuple)):
            record["timings"] = list(timings)
        else:
            record["timings"] = timings
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("select", "capture"))
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--config-json", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if min(args.m, args.n, args.k) <= 0:
        parser.error("M, N and K must be positive")

    import torch
    import flag_gems

    if flag_gems.vendor_name != "kunpeng":
        raise RuntimeError(
            f"expected GEMS_VENDOR=kunpeng before import, got {flag_gems.vendor_name!r}"
        )

    mm_module = importlib.import_module("flag_gems.ops.mm")
    tuner = mm_module.mm_kernel_general
    expected = {(256, 256, 256), (8, 8, 8)}
    observed = {
        (
            int(config.kwargs.get("BLOCK_M", -1)),
            int(config.kwargs.get("BLOCK_N", -1)),
            int(config.kwargs.get("BLOCK_K", -1)),
        )
        for config in tuner.configs
    }
    if not expected.issubset(observed):
        raise RuntimeError(
            "Kunpeng MM does not expose the expected 256^3 and 8^3 configs; "
            f"observed={sorted(observed)}"
        )

    if args.mode == "capture":
        selected = json.loads(args.config_json.read_text(encoding="utf-8"))["selected"]
        wanted = selected["meta"]
        matches = [config for config in tuner.configs if config.kwargs == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"selected config did not match exactly once: {wanted}")
        tuner.configs = matches

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), dtype=dtype, device=flag_gems.device)
    b = torch.randn((args.k, args.n), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.mm(a, b)
    if result.shape != (args.m, args.n) or not torch.isfinite(result).all():
        raise RuntimeError("FlagGems MM produced an invalid result")

    best = tuner.best_config
    if args.mode == "select":
        timing_map = getattr(tuner, "configs_timings", {})
        report = {
            "schema": "kunpeng-mm-selection-v1",
            "shape": {"M": args.m, "N": args.n, "K": args.k},
            "dtype": args.dtype,
            "vendor": flag_gems.vendor_name,
            "selected": config_record(best, timing_map.get(best)),
            "candidates": [
                config_record(config, timing_map.get(config)) for config in tuner.configs
            ],
        }
        args.config_json.parent.mkdir(parents=True, exist_ok=True)
        args.config_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if best.kwargs != wanted:
            raise RuntimeError("capture did not execute the selected config")
        print(f"CAPTURED_CONFIG={json.dumps(config_record(best), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
