"""Controlled component-scoring checks for experimental A5b-N."""
import math
import unittest

from cme_pipeline.stages.tracking.a5b_normalized_scoring import normalized_candidate_score


class TestA5bNormalizedScore(unittest.TestCase):
    def test_all_components_equal_one(self):
        score, terms = normalized_candidate_score(1, 1, 1, 122.5, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(score, 1)
        self.assertTrue(all(x == 1 for x in terms.values()))

    def test_contact_product_and_clipping(self):
        score, terms = normalized_candidate_score(1.5, .8, .5, 61.25, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(terms, {"O": 1., "A": .8, "K": .5, "F": .5})
        self.assertAlmostEqual(score, (.8 * .5 * .5) ** .25)

    def test_noncontact_does_not_collapse_on_zero_overlap(self):
        score, terms = normalized_candidate_score(0, 1, 1, 61.25, 122.5,
                                                  has_spatial_contact=False)
        self.assertEqual(terms["O"], 0.)
        self.assertAlmostEqual(score, .5 ** (1/3))

    def test_negative_and_excessive_terms_bounded(self):
        score, terms = normalized_candidate_score(-1, 5, 3, 400, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(score, 0)
        self.assertTrue(all(0 <= t <= 1 for t in terms.values()))

    def test_invalid_radius_fails(self):
        with self.assertRaises(ValueError):
            normalized_candidate_score(1,1,1,1,0,has_spatial_contact=True)

    def test_equal_factor_interpretation(self):
        score, _ = normalized_candidate_score(.8,.8,.8,98,122.5,
                                               has_spatial_contact=True)
        self.assertAlmostEqual(score, .8)


if __name__ == "__main__":
    unittest.main()
