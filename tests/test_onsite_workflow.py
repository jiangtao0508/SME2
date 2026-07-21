#!/usr/bin/env python3
"""Self-test using synthetic IR text created at runtime.

The fixture is authored for this test and is not copied from a third-party dump.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = (
    ROOT
    / "07_onsite_workflow"
    / "tools"
    / "analyze_onsite_case.py"
)
SUMMARY_TOOL = (
    ROOT
    / "07_onsite_workflow"
    / "tools"
    / "show_onsite_summary.py"
)


class OnsiteWorkflowTest(unittest.TestCase):
    def test_synthetic_multistage_dump(self):
        with tempfile.TemporaryDirectory(prefix="sme2-onsite-test-") as temp:
            base = Path(temp)
            dump = base / "dumps"
            kernel = dump / "bmm_kernel"
            cache = dump / "cache" / "opaque_hash"
            kernel.mkdir(parents=True)
            cache.mkdir(parents=True)

            files = {
                kernel / "00_input.mlir": """
module {
  func.func @demo(%a: memref<32x16xf32>,
                  %b: memref<16x24xf32>,
                  %c: memref<32x24xf32>) {
    linalg.matmul ins(%a, %b : memref<32x16xf32>,
                      memref<16x24xf32>)
                  outs(%c : memref<32x24xf32>)
    return
  }
}
""",
                kernel / "01_after_transform_interpreter.mlir": """
module {
  func.func @demo() {
    scf.for %i = %c0 to %c32 step %c8 {
      vector.contract
    }
    return
  }
}
""",
                kernel / "02_after_earase_schedule.mlir": """
module { func.func @demo() { scf.for %i = %c0 to %c32 step %c8 { vector.outerproduct } return } }
""",
                kernel / "03_after_convert_to_llvm.mlir": """
module {
  func.func @demo() attributes {arm_locally_streaming, arm_new_za} {
    arm_sme.outerproduct
    return
  }
}
""",
                kernel / "04_after_canonicalize.mlir": """
module { func.func @demo() attributes {arm_new_za} { arm_sme.outerproduct return } }
""",
                kernel / "05_after_strip_debug.mlir": """
module { func.func @demo() attributes {arm_new_za} { arm_sme.outerproduct return } }
""",
                kernel / "06_after_legalize_float8.mlir": """
module { func.func @demo() attributes {arm_new_za} { "arm_sme.intr.mopa"() : () -> () return } }
""",
                kernel / "ll.ir": """
define void @demo() #0 {
  call void @llvm.aarch64.sme.mopa.nxv4f32()
  ret void
}
""",
                kernel / "ll.mlir": """
module { llvm.func @demo() attributes {aarch64_new_za} }
""",
                kernel / "tt.mlir": "module { tt.func public @demo() }",
                kernel / "ttshared.mlir": "module { tt.func public @demo() }",
                cache / "bmm_kernel.ttir": "module { tt.func public @demo() }",
                cache / "bmm_kernel.ttsharedir": "module { tt.func public @demo() }",
                cache / "bmm_kernel.llir": (
                    "define void @demo() { "
                    "call void @llvm.aarch64.sme.mopa.nxv4f32() ret void }"
                ),
                dump / "main.cxx": "int main() { return 0; }",
                dump / "kernel.objdump.txt": (
                    "smstart za\nsmstart sm\nfmopa za0.s, p0/m, p0/m, z0.s, z1.s\n"
                    "smstop sm\nsmstop za\n"
                ),
            }
            for path, text in files.items():
                path.write_text(text.strip() + "\n", encoding="utf-8")

            output = base / "result"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--dump",
                    str(dump),
                    "--case-name",
                    "fixture_bmm",
                    "--output",
                    str(output),
                    "--no-package",
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

            chain = json.loads(
                (output / "03_ir_chain_analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(chain["numbered_chain"]), 7)
            self.assertEqual(len(chain["transitions"]), 8)
            self.assertEqual(
                chain["first_seen"]["arm_sme_mlir"]["path"],
                "bmm_kernel/03_after_convert_to_llvm.mlir",
            )
            self.assertIn("llvm_sme", chain["all_text_evidence"])
            self.assertTrue(chain["all_text_evidence"]["llvm_sme"])

            safe = json.loads(
                (output / "07_generic_analysis_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("evidence", safe)
            self.assertNotIn("files", safe)
            report = (output / "ONSITE_REPORT_CN.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("发现 `7` 个带编号的MLIR编译阶段文件", report)
            evidence = (output / "05_sme_evidence.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("dump中已有的反汇编文本", evidence)
            self.assertIn("`mopa`文本计数 `1`", evidence)

            summary = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY_TOOL),
                    "--result",
                    str(output),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(summary.returncode, 0, summary.stdout)
            self.assertIn("现场结论卡", summary.stdout)
            self.assertIn("ArmSME MLIR首次出现", summary.stdout)


if __name__ == "__main__":
    unittest.main()
