#!/usr/bin/env python3
"""Inject the project pass into a copy of Triton CPU's SME Transform schedule."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Optional


SEQUENCE_MARKER = "transform.named_sequence @__transform_main"
BUFFERIZE_INCLUDE = "transform.include @__bufferize_schedule"
SME_INCLUDE = "transform.include @__arm_sme_lowering_schedule"
RESERVED_PREFIX = "%prefetch_"


def find_sequence_end(lines: list[str], start: int) -> int:
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        depth += lines[index].count("{")
        depth -= lines[index].count("}")
        opened = opened or "{" in lines[index]
        if opened and depth == 0:
            return index
    raise ValueError("unterminated @__transform_main")


def inject(
    text: str,
    mode: str,
    snapshot_output: Optional[str],
    argument_index: int,
    distance: int,
    locality: int,
) -> str:
    if argument_index < 0:
        raise ValueError("argument index must be non-negative")
    if distance <= 0:
        raise ValueError("distance must be positive")
    if locality not in range(4):
        raise ValueError("locality must be in the range 0..3")
    if RESERVED_PREFIX in text:
        raise ValueError("input already contains project prefetch SSA names")

    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if SEQUENCE_MARKER in line]
    if len(starts) != 1:
        raise ValueError(
            f"expected exactly one @__transform_main, found {len(starts)}"
        )

    start = starts[0]
    end = find_sequence_end(lines, start)
    bufferized = [
        index for index in range(start, end + 1) if BUFFERIZE_INCLUDE in lines[index]
    ]
    sme = [index for index in range(start, end + 1) if SME_INCLUDE in lines[index]]
    if len(bufferized) != 1 or len(sme) != 1 or bufferized[0] >= sme[0]:
        raise ValueError(
            "expected one ordered bufferize/ArmSME include pair in "
            "@__transform_main"
        )
    match = re.match(
        r"^(?P<indent>\s*)(?P<value>%[-\w.$]+)\s*=",
        lines[bufferized[0]],
    )
    if match is None:
        raise ValueError("could not read the bufferized transform handle")
    indent = match.group("indent")
    module_handle = match.group("value")
    if mode == "snapshot":
        if not snapshot_output:
            raise ValueError("snapshot mode requires --snapshot-output")
        if any(character.isspace() for character in snapshot_output):
            raise ValueError("snapshot output path must not contain whitespace")
        pass_name = "prefetch-snapshot"
        options_attribute = f' {{options = "output-path={snapshot_output}"}}'
    else:
        pass_name = (
            "prefetch-gemm-rhs" if mode == "gemm-rhs" else "prefetch-materialize"
        )
        options = (
            f"distance={distance} locality={locality}"
        )
        if mode == "materialize":
            options = f"argument-index={argument_index} {options}"
        options_attribute = f' {{options = "{options}"}}'

    inserted = [
        f'{indent}%prefetch_funcs = transform.structured.match ops{{["func.func"]}} '
        f'in {module_handle} : (!transform.any_op) -> !transform.op<"func.func">\n',
        f'{indent}%prefetch_functions = transform.apply_registered_pass '
        f'"{pass_name}" to %prefetch_funcs{options_attribute} '
        f': (!transform.op<"func.func">) -> !transform.op<"func.func">\n',
    ]
    lines[sme[0] : sme[0]] = inserted
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--mode",
        choices=("snapshot", "materialize", "gemm-rhs"),
        default="snapshot",
    )
    parser.add_argument("--snapshot-output")
    parser.add_argument("--argument-index", type=int, default=0)
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--locality", type=int, default=3)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ; the source dump is read-only")

    try:
        result = inject(
            args.input.read_text(),
            args.mode,
            args.snapshot_output,
            args.argument_index,
            args.distance,
            args.locality,
        )
    except (OSError, ValueError) as error:
        print(f"inject_schedule.py: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f"wrote modified copy: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
