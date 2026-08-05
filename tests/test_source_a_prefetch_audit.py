#!/usr/bin/env python3
"""Tests for the fail-closed source-A prefetch text audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "prefetch_plugin" / "audit_bmm_source_prefetch.py"


def candidate(prefetch_target: str) -> str:
    prefetches = "\n".join(
        f"memref.prefetch {prefetch_target}[%r{row}, %c0], read, "
        "locality<2>, data : memref<4x4xbf16>"
        for row in range(4)
    )
    return f"""
%future = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [4, 4], strides: [%stride, 1] {{prefetch.distance_iterations = 8 : i64, prefetch.source_argument = 0 : i64}} : memref<*xbf16> to memref<4x4xbf16>
{prefetches}
}} {{prefetch.issue_every = 8 : i64}}
memref.copy %current, %alloc : memref<4x4xbf16> to memref<4x4xbf16>
"""


class SourceAPrefetchAuditTest(unittest.TestCase):
    def run_audit(self, text: str) -> tuple[subprocess.CompletedProcess, dict]:
        with tempfile.TemporaryDirectory(prefix="sme2-source-a-audit-") as temp:
            base = Path(temp)
            input_path = base / "candidate.mlir"
            report_path = base / "audit.json"
            input_path.write_text(text)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT),
                    str(input_path),
                    "--source-argument",
                    "0",
                    "--expected-prefetches",
                    "4",
                    "--distance",
                    "8",
                    "--issue-every",
                    "8",
                    "--locality",
                    "2",
                    "--json",
                    str(report_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return completed, json.loads(report_path.read_text())

    def test_accepts_original_source_future_view(self):
        completed, report = self.run_audit(candidate("%future"))
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(report["status"], "passed")

    def test_rejects_private_alloc_prefetch(self):
        completed, report = self.run_audit(candidate("%alloc"))
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["no_private_alloc_prefetch_target"])


if __name__ == "__main__":
    unittest.main()
