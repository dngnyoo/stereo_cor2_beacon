"""Unit tests for per-track A5b comparison geometry (no live datasets)."""
import unittest
import numpy as np

from cme_pipeline.tracking_comparison import _frame_tracks


class TestTrackingGeometry(unittest.TestCase):
    def test_disconnected_islands_belonging_to_one_track_produce_one_geometry(self):
        owners = np.zeros((64, 64), dtype=np.int32)
        owners[8:10, 20:22] = 27
        owners[35:37, 40:42] = 27
        y, x = np.indices((64, 64))
        pa = np.rad2deg(np.arctan2(-(x * 4 + 1.5 - 127.5),
                                   y * 4 + 1.5 - 127.5)) % 360
        radius = np.hypot(x * 4 + 1.5 - 127.5, y * 4 + 1.5 - 127.5)
        tracks = _frame_tracks(owners, pa, radius)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_id"], 27)
        self.assertEqual(tracks[0]["n_patches"], 8)
        self.assertEqual(int(tracks[0]["mask"].sum()), 8)

    def test_two_tracks_remain_separate(self):
        owners = np.zeros((64, 64), dtype=np.int32)
        owners[10, 10] = 1
        owners[15, 15] = 2
        pa = np.zeros((64, 64))
        radius = np.ones((64, 64))
        tracks = _frame_tracks(owners, pa, radius)
        self.assertEqual([x["track_id"] for x in tracks], [1, 2])


if __name__ == "__main__":
    unittest.main()
