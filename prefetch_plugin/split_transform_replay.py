#!/usr/bin/env python3
"""Split an SME Transform schedule around bufferization for offline rewriting."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


MAIN_MARKER = "transform.named_sequence @__transform_main"
BUFFERIZE_INCLUDE = "transform.include @__bufferize_schedule"
SME_INCLUDE = "transform.include @__arm_sme_lowering_schedule"
SCHEDULE_MARKER = "module attributes {transform.with_named_sequence}"


def find_block_end(lines: list[str], start: int, name: str) -> int:
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        depth += lines[index].count("{")
        depth -= lines[index].count("}")
        opened = opened or "{" in lines[index]
        if opened and depth == 0:
            return index
    raise ValueError(f"unterminated {name}")


def find_unique(lines: list[str], marker: str, begin: int, end: int) -> int:
    matches = [index for index in range(begin, end) if marker in lines[index]]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {marker!r}, found {len(matches)}")
    return matches[0]


def locate_main(lines: list[str]) -> tuple[int, int]:
    start = find_unique(lines, MAIN_MARKER, 0, len(lines))
    return start, find_block_end(lines, start, "@__transform_main")


def make_prefix(original: str) -> str:
    lines = original.splitlines(keepends=True)
    main_start, main_end = locate_main(lines)
    bufferize = find_unique(lines, BUFFERIZE_INCLUDE, main_start, main_end)
    indent_match = re.match(r"^(\s*)", lines[bufferize])
    assert indent_match is not None
    indent = indent_match.group(1)
    lines[bufferize + 1 : main_end] = [f"{indent}transform.yield\n"]
    return "".join(lines)


def add_tptr_preload(payload: str) -> str:
    # Some triton-shared builds register the tptr dialect lazily from the
    # tptr-to-llvm pass. MLIR does not allow the first dialect load from a
    # multi-threaded pass execution context. Keep one harmless tptr op in the
    # payload so parsing loads the dialect before the pass manager starts.
    if "tptr.type_offset" in payload:
        return payload
    payload_lines = payload.splitlines(keepends=True)
    for index, line in enumerate(payload_lines):
        match = re.match(r"^(\s*)func\.func\b.*\{\s*$", line)
        if match:
            indent = match.group(1) + "  "
            payload_lines.insert(
                index + 1,
                f"{indent}%__prefetch_preload_tptr = tptr.type_offset f32\n",
            )
            return "".join(payload_lines)
    raise ValueError("could not find a func.func body for tptr preload")


def normalize_location_spacing(text: str) -> str:
    """Restore required whitespace before MLIR location suffixes."""
    for boundary in ("}", ")", "]"):
        text = text.replace(f"{boundary}loc(", f"{boundary} loc(")
    return text


def make_resume(original: str, payload: str) -> str:
    payload = add_tptr_preload(payload)

    original_lines = original.splitlines(keepends=True)
    schedule_start = find_unique(
        original_lines, SCHEDULE_MARKER, 0, len(original_lines)
    )
    schedule_lines = original_lines[schedule_start:]
    main_start, main_end = locate_main(schedule_lines)
    sme = find_unique(schedule_lines, SME_INCLUDE, main_start, main_end)

    main_signature = schedule_lines[main_start]
    root_match = re.search(r"\((%[-\w.$]+):\s*!transform", main_signature)
    if root_match is None:
        raise ValueError("could not identify @__transform_main root handle")
    root_handle = root_match.group(1)

    sme_match = re.search(r"\((%[-\w.$]+)\)", schedule_lines[sme])
    if sme_match is None:
        raise ValueError("could not identify ArmSME input handle")
    module_handle = sme_match.group(1)

    indent_match = re.match(r"^(\s*)", schedule_lines[sme])
    assert indent_match is not None
    indent = indent_match.group(1)
    resume_prefix = [
        f'{indent}%prefetch_resume_funcs = transform.structured.match '
        f'ops{{["func.func"]}} in {root_handle} : (!transform.any_op) '
        f'-> !transform.any_op\n',
        f"{indent}{module_handle} = transform.get_parent_op "
        f"%prefetch_resume_funcs {{deduplicate}} : (!transform.any_op) "
        f"-> !transform.any_op\n",
    ]
    schedule_lines[main_start + 1 : sme] = resume_prefix
    schedule = "".join(schedule_lines)
    return normalize_location_spacing(payload.rstrip() + "\n" + schedule)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefix = subparsers.add_parser("prefix")
    prefix.add_argument("original", type=pathlib.Path)
    prefix.add_argument("output", type=pathlib.Path)

    preload = subparsers.add_parser("preload")
    preload.add_argument("original", type=pathlib.Path)
    preload.add_argument("output", type=pathlib.Path)

    resume = subparsers.add_parser("resume")
    resume.add_argument("original", type=pathlib.Path)
    resume.add_argument("payload", type=pathlib.Path)
    resume.add_argument("output", type=pathlib.Path)

    args = parser.parse_args()
    output = args.output
    inputs = [args.original]
    if args.command == "resume":
        inputs.append(args.payload)
    if any(output.resolve() == input_path.resolve() for input_path in inputs):
        parser.error("output must differ from every input")

    try:
        original = args.original.read_text()
        if args.command == "prefix":
            result = make_prefix(original)
        elif args.command == "preload":
            result = add_tptr_preload(original)
        else:
            result = make_resume(original, args.payload.read_text())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result)
    except (OSError, ValueError) as error:
        print(f"split_transform_replay.py: {error}", file=sys.stderr)
        return 1

    print(f"wrote {args.command} replay file: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
