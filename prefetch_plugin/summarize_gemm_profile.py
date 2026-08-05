#!/usr/bin/env python3
"""Print one compact numeric lineage line per GEMM candidate."""

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <GemmKernelProfile.v1.json>", file=sys.stderr)
        return 2
    try:
        profile = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
        print(f"candidate_count={profile['candidate_count']}")
        for candidate in profile["candidates"]:
            lineage = candidate["lineage"]
            arguments = ",".join(str(value) for value in lineage["source_argument_indices"]) or "none"
            print(
                f"candidate={candidate['candidate_id']} "
                f"writers={lineage['writer_operation_count']} "
                f"copy={lineage['memref_copy_writer_count']} "
                f"vector_write={lineage['vector_transfer_write_count']} "
                f"store={lineage['memref_store_writer_count']} "
                f"linalg={lineage['linalg_writer_count']} "
                f"source_args={arguments}"
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"summarize_gemm_profile.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

