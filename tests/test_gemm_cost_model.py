import json
from pathlib import Path
import unittest

from cost_model.plan_gemm_rhs import derive_plan


def hardware_profile():
    return {
        "schema_version": "1.1",
        "profile_id": "measured-test",
        "source": "measured",
        "derived": {
            "cache_line_bytes": 64,
            "l1d_bytes": 65536,
            "l2_bytes": 2097152,
            "memory_latency_ns": 100.0,
            "sustainable_bandwidth_bytes_per_ns": 32.0,
            "prefetch_instruction_cost_ns": 0.25,
            "max_outstanding_prefetches": 32,
        },
        "quality": {"prefetch_instruction_verified_in_assembly": True, "warnings": []},
    }


def kernel_profile(trip_count=64):
    return {
        "schema_version": "1.0",
        "candidates": [
            {
                "candidate_id": 0,
                "loop_trip_count": trip_count,
                "source_allocation_bytes": 16384,
                "rhs_row_bytes": 512,
                "vector_read_bytes": 128,
            }
        ],
    }


class GemmCostModelTests(unittest.TestCase):
    def test_plan_schema_is_valid_json(self):
        schema = Path(__file__).resolve().parents[1] / "cost_model" / "prefetch-plan-v1.1.schema.json"
        self.assertEqual(
            "Formula-driven packed GEMM RHS PrefetchPlan 1.1",
            json.loads(schema.read_text(encoding="utf-8"))["title"],
        )

    def test_derives_formula_driven_plan(self):
        plan = derive_plan(hardware_profile(), kernel_profile(), anchor_step_ns=25.0)
        decision = plan["decisions"][0]
        self.assertEqual(4, decision["distance"]["value"])
        self.assertEqual(2, decision["emission"]["coverage_lines"])
        self.assertEqual(1, decision["emission"]["issue_every"])
        self.assertTrue(plan["model_config"]["no_parameter_search"])

    def test_measured_issue_cost_can_throttle_frequency(self):
        hardware = hardware_profile()
        hardware["derived"]["prefetch_instruction_cost_ns"] = 4.0
        plan = derive_plan(hardware, kernel_profile(), anchor_step_ns=10.0)
        self.assertGreater(plan["decisions"][0]["emission"]["issue_every"], 1)

    def test_short_loop_yields_explicit_no_prefetch(self):
        plan = derive_plan(hardware_profile(), kernel_profile(trip_count=4), anchor_step_ns=25.0)
        self.assertEqual([], plan["decisions"])
        self.assertEqual("REQUIRED_DISTANCE_EXCEEDS_K_LOOP", plan["diagnostics"]["rejection"])

    def test_rejects_fallback_hardware(self):
        hardware = hardware_profile()
        hardware["source"] = "fallback"
        with self.assertRaises(ValueError):
            derive_plan(hardware, kernel_profile(), anchor_step_ns=25.0)


if __name__ == "__main__":
    unittest.main()
