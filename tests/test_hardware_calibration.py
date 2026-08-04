import json
from pathlib import Path
import tempfile
import unittest

from hardware_calibration.calibrate_hardware import (
    aggregate_probe_runs,
    derive_profile,
    discover_linux_cache,
    parse_probe_output,
    parse_size,
)


SAMPLE_OUTPUT_1 = """\
latency\t4096\t1.0
latency\t32768\t2.0
latency\t262144\t6.0
latency\t4194304\t20.0
bandwidth\t67108864\t3\t32.0
stride\t64\t4.0
stride\t128\t4.5
stride\t256\t8.0
prefetch\t0\t1.0
prefetch\t1\t1.2
prefetch\t2\t1.1
prefetch\t4\t1.05
"""

SAMPLE_OUTPUT_2 = """\
latency\t4096\t1.2
latency\t32768\t2.2
latency\t262144\t6.2
latency\t4194304\t22.0
bandwidth\t67108864\t3\t30.0
stride\t64\t4.2
stride\t128\t4.7
stride\t256\t8.5
prefetch\t0\t1.0
prefetch\t1\t1.3
prefetch\t2\t1.15
prefetch\t4\t1.075
"""


class HardwareCalibrationTests(unittest.TestCase):
    def test_parse_size(self) -> None:
        self.assertEqual(64, parse_size("64"))
        self.assertEqual(64 * 1024, parse_size("64K"))
        self.assertEqual(2 * 1024 * 1024, parse_size("2M"))
        self.assertIsNone(parse_size("bad"))

    def test_parse_and_aggregate_probe_output(self) -> None:
        first = parse_probe_output(SAMPLE_OUTPUT_1)
        second = parse_probe_output(SAMPLE_OUTPUT_2)
        aggregate = aggregate_probe_runs([first, second])
        latency_4k = next(point for point in aggregate["latency"] if point["key"] == 4096)
        self.assertAlmostEqual(1.1, latency_4k["median"])
        self.assertAlmostEqual(31.0, aggregate["bandwidth"]["bytes_per_ns_median"])

    def test_discover_linux_cache_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index0"
            index.mkdir()
            values = {
                "level": "1\n",
                "type": "Data\n",
                "size": "64K\n",
                "coherency_line_size": "64\n",
                "ways_of_associativity": "4\n",
                "number_of_sets": "256\n",
                "shared_cpu_list": "0\n",
            }
            for name, value in values.items():
                (index / name).write_text(value, encoding="utf-8")
            caches = discover_linux_cache(root)
        self.assertEqual(1, len(caches))
        self.assertEqual(65536, caches[0]["size_bytes"])
        self.assertEqual(64, caches[0]["line_bytes"])

    def test_derive_profile_keeps_measured_units(self) -> None:
        aggregate = aggregate_probe_runs(
            [parse_probe_output(SAMPLE_OUTPUT_1), parse_probe_output(SAMPLE_OUTPUT_2)]
        )
        caches = [
            {
                "level": 1,
                "type": "Data",
                "size_bytes": 32768,
                "line_bytes": 64,
                "ways": 4,
                "sets": 128,
                "shared_cpu_list": "0",
            },
            {
                "level": 2,
                "type": "Unified",
                "size_bytes": 1048576,
                "line_bytes": 64,
                "ways": 8,
                "sets": 2048,
                "shared_cpu_list": "0",
            },
        ]
        derived, warnings = derive_profile(aggregate, caches, frequency_ghz=2.5)
        self.assertEqual(64, derived["cache_line_bytes"])
        self.assertEqual(32768, derived["l1d_bytes"])
        self.assertAlmostEqual(21.0, derived["memory_latency_ns"])
        self.assertAlmostEqual(52.5, derived["memory_latency_cycles_estimate"])
        self.assertGreater(derived["prefetch_instruction_cost_ns"], 0)
        self.assertIsInstance(warnings, list)

    def test_schema_is_valid_json(self) -> None:
        schema = Path(__file__).resolve().parents[1] / "hardware_calibration" / "hardware-profile-v1.1.schema.json"
        parsed = json.loads(schema.read_text(encoding="utf-8"))
        self.assertEqual("SME Prefetch HardwareProfile 1.1", parsed["title"])


if __name__ == "__main__":
    unittest.main()
