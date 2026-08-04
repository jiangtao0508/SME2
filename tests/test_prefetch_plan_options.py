import unittest

from prefetch_plugin.prefetch_plan_options import resolve


def plan(version="1.1", issue_every=2):
    return {
        "schema_version": version,
        "hardware_profile": {"cache_line_bytes": 64},
        "decisions": [
            {
                "decision_id": "rhs",
                "object_id": "packed_rhs_panel",
                "strategy": "TILE_PREFETCH",
                "distance": {"value": 4, "unit": "ITERATION"},
                "target_cache": "L2",
                "granularity": {"kind": "PANEL", "bytes": 128},
                "emission": {"issue_every": issue_every, "coverage_lines": 2},
            }
        ],
    }


class PrefetchPlanOptionsTests(unittest.TestCase):
    def test_v11_emission_reaches_plugin_options(self):
        values = resolve(plan())
        self.assertEqual((4, 2, 2, 2, 64, "rhs", "packed_rhs_panel"), values)

    def test_v10_without_emission_defaults_to_every_iteration(self):
        value = plan(version="1.0")
        del value["decisions"][0]["emission"]
        self.assertEqual(1, resolve(value)[3])

    def test_rejects_inconsistent_coverage(self):
        value = plan()
        value["decisions"][0]["emission"]["coverage_lines"] = 1
        with self.assertRaises(ValueError):
            resolve(value)


if __name__ == "__main__":
    unittest.main()
