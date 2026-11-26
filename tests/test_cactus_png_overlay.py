import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.visualize_cactus_overlay import active_events, load_cactus, overlay_frame


class TestCactusOverlay(unittest.TestCase):
    def test_active_interval_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "ref.csv"
            pd.DataFrame([
                dict(catalog_event_id="CACTUS_CME_0001", event_type="CME",
                     onset_utc="2024-05-03T09:00:00Z", liftoff_duration_hours=2,
                     cpa_deg=0, angular_width_deg=20),
                dict(catalog_event_id="CACTUS_FLOW_0001", event_type="Flow",
                     onset_utc="2024-05-03T09:00:00Z", liftoff_duration_hours=2,
                     cpa_deg=90, angular_width_deg=20),
            ]).to_csv(csv, index=False)
            ref = load_cactus(csv)
            self.assertEqual(len(active_events(ref, pd.Timestamp("2024-05-03T11:00:00Z"))), 1)
            self.assertTrue(active_events(ref, pd.Timestamp("2024-05-03T11:00:01Z")).empty)
            source = root / "pair_001_20240503T093000Z_families.png"
            Image.new("RGB", (864, 840), "black").save(source)
            original = source.read_bytes()
            dest = root / "out.png"
            self.assertEqual(overlay_frame(source, dest, ref, sx=128, sy=128), 1)
            self.assertEqual(source.read_bytes(), original)
            with Image.open(dest) as im:
                self.assertEqual(im.size, (864, 840))
                self.assertGreater(im.getpixel((432, 200))[0], 0)
            with self.assertRaises(ValueError):
                overlay_frame(source, source, ref, sx=128, sy=128)
