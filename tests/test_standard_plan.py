import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_plan import build_plan


class StandardPlanTests(unittest.TestCase):
    def test_plan_and_viz_override(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "example.yaml"
            f.write_text('experiment:\n  date: "20240503"\ntracking:\n  method: a5b\n  visualization: {mode: diagnostic}\n')
            c = StandardConfig.load(f)
            plan = build_plan(c, "gates")
            self.assertEqual(len(plan["stages"]), 5)
            self.assertEqual(plan["stages"][2]["visualization"], "diagnostic")
            forced = build_plan(c, "gates", "off")
            self.assertTrue(all(x["visualization"] == "off" for x in forced["stages"]))

    def test_yaml_off_boolean_and_quoted(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "unquoted.yaml"
            f.write_text('experiment:\n  date: "20240503"\nvisualization:\n  default: off\npreprocessing:\n  visualization: {mode: off}\n')
            c = StandardConfig.load(f)
            self.assertEqual(c.vis_mode("preprocessing"), "off")
            self.assertEqual(c.vis_mode("tracking"), "off")
            f.write_text('experiment:\n  date: "20240503"\nvisualization:\n  default: "off"\n')
            c = StandardConfig.load(f)
            self.assertEqual(c.vis_mode("tracking"), "off")

    def test_yaml_on_boolean_rejected(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "invalid.yaml"
            f.write_text('experiment:\n  date: "20240503"\nvisualization:\n  default: on\n')
            with self.assertRaises(ValueError):
                StandardConfig.load(f)

    def test_unknown_key_rejected(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "bad.yaml"
            f.write_text('experiment:\n  date: "20240503"\nmisspelled_tracking: {}\n')
            with self.assertRaises(ValueError):
                StandardConfig.load(f)


if __name__ == "__main__":
    unittest.main()
