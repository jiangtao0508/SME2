import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).parents[1] / "prefetch_plugin" / "onsite_mm_select.py"
SPEC = importlib.util.spec_from_file_location("onsite_mm_select", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Wrapper:
    def __init__(self, fn):
        self.fn = fn


class FakeTuner:
    def __init__(self, configs=None):
        self.configs = configs or [object()]

    def run(self):
        pass


class OnsiteMmSelectTest(unittest.TestCase):
    def test_unwraps_libentry_to_tuner(self):
        tuner = FakeTuner()
        self.assertIs(MODULE.find_autotuner(Wrapper(Wrapper(tuner))), tuner)

    def test_reports_wrapper_chain_when_tuner_is_missing(self):
        with self.assertRaisesRegex(RuntimeError, "Wrapper"):
            MODULE.find_autotuner(Wrapper(object()))

    def test_force_selected_config_keeps_exact_match(self):
        class Config:
            def __init__(self, kwargs):
                self.kwargs = kwargs

        small = Config({"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 8})
        large = Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256})
        tuner = FakeTuner([small, large])
        selected = MODULE.force_selected_config(
            tuner,
            {"selected": {"meta": dict(large.kwargs)}},
        )
        self.assertIs(selected, large)
        self.assertEqual(tuner.configs, [large])

    def test_force_selected_config_rejects_missing_match(self):
        class Config:
            kwargs = {"BLOCK_M": 8}

        with self.assertRaisesRegex(RuntimeError, "did not match exactly once"):
            MODULE.force_selected_config(
                FakeTuner([Config()]),
                {"selected": {"meta": {"BLOCK_M": 256}}},
            )


if __name__ == "__main__":
    unittest.main()
