import unittest

from sme_timing.calibrate_sme import derive_profile, parse_probe_output


SAMPLE = """\
system\tstreaming_vector_bytes\t64
system\ttimer\tCLOCK_MONOTONIC_RAW
timing\tbaseline\t0\t1000\t100\t100000000
timing\tbaseline\t0\t1000\t110\t100000000
timing\tfmopa_one_tile\t1\t1000\t5100\t100000000
timing\tfmopa_one_tile\t1\t1000\t5110\t100000000
timing\tfmopa_four_tiles\t4\t1000\t8100\t100000000
timing\tfmopa_four_tiles\t4\t1000\t8110\t100000000
"""


class SmeTimingTests(unittest.TestCase):
    def test_parse_and_subtract_loop_baseline(self):
        profile = derive_profile(parse_probe_output(SAMPLE))
        self.assertEqual(64, profile["streaming_vector_bytes"])
        self.assertEqual(512, profile["f32_flops_per_fmopa"])
        self.assertAlmostEqual(50.0, profile["one_tile_dependency_ns_per_fmopa"])
        self.assertAlmostEqual(20.0, profile["four_tile_throughput_ns_per_fmopa"])

    def test_incomplete_output_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_probe_output("system\tstreaming_vector_bytes\t64\n")


if __name__ == "__main__":
    unittest.main()
