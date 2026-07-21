#!/usr/bin/env python3
"""End-to-end test with a synthetic pytest module and fake compiler."""

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = (
    ROOT
    / "07_onsite_workflow"
    / "tools"
    / "trace_pytest_pipeline.py"
)
SUMMARY_TOOL = (
    ROOT
    / "07_onsite_workflow"
    / "tools"
    / "show_onsite_summary.py"
)
SPEC = importlib.util.spec_from_file_location("trace_pytest_pipeline", TOOL)
TRACE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE_MODULE)


class PytestPipelineTraceTest(unittest.TestCase):
    def test_collect_failure_does_not_execute_test(self):
        with tempfile.TemporaryDirectory(prefix="sme2-collect-fail-") as temp:
            base = Path(temp)
            workdir = base / "project"
            fake_pytest = workdir / "pytest"
            output = base / "report"
            marker = base / "unexpected_run.txt"
            fake_pytest.mkdir(parents=True)
            (fake_pytest / "__init__.py").write_text("", encoding="utf-8")
            (fake_pytest / "__main__.py").write_text(
                textwrap.dedent(
                    """
                    import os
                    import sys
                    from pathlib import Path

                    if "--collect-only" in sys.argv:
                        print("ERROR: test node not found")
                        raise SystemExit(4)
                    Path(os.environ["RUN_MARKER"]).write_text("executed")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(TOOL),
                "--workdir",
                str(workdir),
                "--nodeid",
                "tests/missing.py::test_missing[case0]",
                "--watch-root",
                str(base / "dumps"),
                "--output",
                str(output),
                "--strace",
                "off",
                "--env",
                "PYTHONPATH={}".format(workdir),
                "--env",
                "RUN_MARKER={}".format(marker),
                "--run",
            ]
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 4, completed.stdout)
            self.assertFalse(marker.exists())
            report = (output / "PYTEST_PIPELINE_REPORT_CN.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("pytest收集失败", report)

    def test_strace_writer_is_mapped_to_exec(self):
        with tempfile.TemporaryDirectory(prefix="sme2-strace-parse-") as temp:
            base = Path(temp)
            trace_dir = base / "trace"
            dump = base / "dumps"
            trace_dir.mkdir()
            dump.mkdir()
            artifact = dump / "00_input.mlir"
            (trace_dir / "trace.4312").write_text(
                '1784600000.100000 execve("/opt/compiler/bin/custom-cc", '
                '["custom-cc", "--emit-mlir"], 0x0 /* 0 vars */) = 0 <0.001000>\n'
                '1784600000.200000 openat(AT_FDCWD, "{}", '
                'O_WRONLY|O_CREAT|O_TRUNC, 0666) = 3 <0.000100>\n'.format(
                    artifact
                ),
                encoding="utf-8",
            )
            parsed = TRACE_MODULE.parse_strace(
                trace_dir, base, [dump.resolve()]
            )
            self.assertEqual(len(parsed["exec_events"]), 1)
            self.assertEqual(len(parsed["write_events"]), 1)
            writer = parsed["write_events"][0]
            self.assertEqual(writer["pid"], 4312)
            self.assertEqual(writer["path"], str(artifact.resolve()))
            self.assertIn("custom-cc", writer["producer_exec"])

    def test_synthetic_pytest_and_compiler_trace(self):
        with tempfile.TemporaryDirectory(prefix="sme2-pytest-trace-") as temp:
            base = Path(temp)
            workdir = base / "project"
            tests_dir = workdir / "tests"
            fake_pytest = workdir / "pytest"
            dump = base / "dumps"
            output = base / "report"
            tests_dir.mkdir(parents=True)
            fake_pytest.mkdir()

            nodeid = "tests/test_demo.py::test_demo[shape0]"
            (tests_dir / "test_demo.py").write_text(
                textwrap.dedent(
                    """
                    import flag_gems
                    import torch

                    @demo.parametrize("shape", [(32, 16, 24)])
                    def test_demo(shape):
                        with flag_gems.use_gems():
                            return torch.bmm(shape, shape)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            compiler = workdir / "fake_compiler.py"
            compiler.write_text(
                textwrap.dedent(
                    """
                    import os
                    import time
                    from pathlib import Path

                    root = Path(os.environ["FAKE_DUMP"])
                    kernel = root / "demo_kernel"
                    kernel.mkdir(parents=True, exist_ok=True)
                    stages = [
                        ("00_input.mlir", "module { linalg.matmul }"),
                        (
                            "01_after_transform_interpreter.mlir",
                            "module { scf.for vector.outerproduct }",
                        ),
                        (
                            "03_after_convert_to_llvm.mlir",
                            "module { arm_sme.outerproduct }",
                        ),
                    ]
                    for name, text in stages:
                        (kernel / name).write_text(text + "\\n", encoding="utf-8")
                        time.sleep(0.08)
                    (root / "main.cxx").write_text(
                        "int main() { return 0; }\\n", encoding="utf-8"
                    )
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (fake_pytest / "__init__.py").write_text("", encoding="utf-8")
            (fake_pytest / "__main__.py").write_text(
                textwrap.dedent(
                    """
                    import os
                    import subprocess
                    import sys

                    nodeid = sys.argv[-1]
                    if "--collect-only" in sys.argv:
                        print("PLUGIN registered: synthetic-demo-plugin")
                        print(nodeid)
                        raise SystemExit(0)
                    completed = subprocess.run(
                        [sys.executable, os.environ["FAKE_COMPILER"]],
                        check=False,
                    )
                    print("synthetic pytest completed")
                    raise SystemExit(completed.returncode)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            pythonpath = str(workdir)
            existing = os.environ.get("PYTHONPATH")
            if existing:
                pythonpath += os.pathsep + existing
            command = [
                sys.executable,
                str(TOOL),
                "--workdir",
                str(workdir),
                "--nodeid",
                nodeid,
                "--watch-root",
                str(dump),
                "--dump",
                str(dump),
                "--scan-root",
                str(workdir),
                "--output",
                str(output),
                "--strace",
                "off",
                "--interval-ms",
                "20",
                "--env",
                "PYTHONPATH={}".format(pythonpath),
                "--env",
                "FAKE_DUMP={}".format(dump),
                "--env",
                "FAKE_COMPILER={}".format(compiler),
                "--run",
            ]
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

            ast_info = json.loads(
                (output / "02_test_ast.json").read_text(encoding="utf-8")
            )
            self.assertIn("torch.bmm", ast_info["calls"])
            self.assertIn("flag_gems.use_gems", ast_info["with_contexts"])

            collect = json.loads(
                (output / "01_pytest_collect.json").read_text(encoding="utf-8")
            )
            self.assertEqual(collect["exit_code"], 0)
            self.assertTrue(collect["plugin_lines"])

            events = json.loads(
                (output / "05_file_timeline.json").read_text(encoding="utf-8")
            )
            created_names = {
                Path(row["path"]).name
                for row in events
                if row["event"] == "created"
            }
            self.assertIn("00_input.mlir", created_names)
            self.assertIn("03_after_convert_to_llvm.mlir", created_names)

            matches = json.loads(
                (output / "08_source_name_matches.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(matches["matches"])
            self.assertTrue(
                (output / "artifact_analysis" / "ONSITE_REPORT_CN.md").is_file()
            )
            report = (output / "PYTEST_PIPELINE_REPORT_CN.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("pytest到中间文件完整追踪报告", report)
            self.assertIn("追踪会引入明显开销", report)

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
            self.assertIn("pytest：收集成功；运行成功", summary.stdout)
            self.assertIn("torch.bmm=有", summary.stdout)
            self.assertIn("00_input.mlir", summary.stdout)
            self.assertIn("ArmSME MLIR首次出现", summary.stdout)


if __name__ == "__main__":
    unittest.main()
