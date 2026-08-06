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
    def __init__(self):
        self.configs = [object()]

    def run(self):
        pass


class OnsiteMmSelectTest(unittest.TestCase):
    def test_unwraps_libentry_to_tuner(self):
        tuner = FakeTuner()
        self.assertIs(MODULE.find_autotuner(Wrapper(Wrapper(tuner))), tuner)

    def test_reports_wrapper_chain_when_tuner_is_missing(self):
        with self.assertRaisesRegex(RuntimeError, "Wrapper"):
            MODULE.find_autotuner(Wrapper(object()))


if __name__ == "__main__":
    unittest.main()
