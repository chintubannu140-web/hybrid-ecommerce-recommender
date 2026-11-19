import math
import unittest

from recommender import Recommender


class RankingTests(unittest.TestCase):
    def setUp(self):
        catalog = {
            "shoes": "trail shoes", "belt": "trail belt",
            "bottle": "glass bottle", "socks": "trail socks",
        }
        baskets = {
            "shopper": ["shoes", "belt", "belt"],
            "runner": ["belt", "socks"],
            "walker": ["shoes", "bottle"],
            "guest": ["shoes"],
        }
        self.model = Recommender(catalog, [
            {"user_id": user, "product_id": item}
            for user, items in baskets.items() for item in items
        ])

    def test_purchase_vectors_use_frequency_and_distinct_buyer_counts(self):
        # Three shoppers bought shoes and two bought belts. This shopper bought two belts.
        vector = self.model.user_vectors["shopper"]
        shoe_weight = 1 + math.log(5 / 4)
        belt_weight = (1 + math.log(2)) * (1 + math.log(5 / 3))
        self.assertAlmostEqual(vector["belt"] / vector["shoes"], belt_weight / shoe_weight)

    def test_hybrid_uses_both_scores_and_excludes_prior_purchases(self):
        signals = self.model.signals("shopper")
        self.assertGreater(signals["socks"][0], signals["bottle"][0])
        self.assertGreater(signals["socks"][1], signals["bottle"][1])
        ranked = self.model.recommend("shopper")
        self.assertEqual([item for item, _, _ in ranked], ["socks", "bottle"])
        for item, score, _ in ranked:
            collaborative, content = signals[item]
            self.assertAlmostEqual(score, 0.6 * collaborative + 0.4 * content)

    def test_cold_start_counts_every_purchase(self):
        self.assertEqual(self.model.recommend("new", limit=2), [
            ("belt", 3, "popularity"), ("shoes", 3, "popularity"),
        ])
