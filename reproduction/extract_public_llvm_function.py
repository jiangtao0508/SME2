#!/usr/bin/env python3
"""Run compiler.py's public _extract_mlir_function method without importing Triton."""

import argparse
import ast
from pathlib import Path
import re
import textwrap


def extract_method(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CPUBackend"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_extract_mlir_function"
    )
    lines = source.splitlines(keepends=True)
    method_source = textwrap.dedent("".join(lines[method.lineno - 1 : method.end_lineno]))
    namespace = {"re": re, "textwrap": textwrap}
    exec(compile(method_source, str(source_path), "exec"), namespace)
    return namespace["_extract_mlir_function"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    extract_method(args.compiler)(object(), str(args.input))


if __name__ == "__main__":
    main()
