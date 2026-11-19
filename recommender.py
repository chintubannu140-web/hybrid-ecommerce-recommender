"""Recommend products from purchase history and product descriptions."""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_data(products_path=DATA / "products.csv", purchases_path=DATA / "purchases.csv"):
    with products_path.open(newline="", encoding="utf-8") as source:
        products = {row["product_id"]: row for row in csv.DictReader(source)}
    with purchases_path.open(newline="", encoding="utf-8") as source:
        purchases = list(csv.DictReader(source))
    missing = {row["product_id"] for row in purchases} - products.keys()
    if missing:
        raise ValueError(f"unknown product IDs in purchases: {', '.join(sorted(missing))}")
    catalog = {
        item: " ".join(row[field] for field in ("name", "category", "description"))
        for item, row in products.items()
    }
    return products, catalog, purchases


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / length for key, value in vector.items()} if length else {}


def cosine(left, right):
    return sum(value * right.get(key, 0) for key, value in left.items())


def product_vectors(catalog):
    counts = {
        item: Counter(re.findall(r"[a-z0-9]+", text.lower()))
        for item, text in catalog.items()
    }
    document_frequency = Counter(word for terms in counts.values() for word in terms)
    idf = {
        word: 1 + math.log((1 + len(counts)) / (1 + frequency))
        for word, frequency in document_frequency.items()
    }
    return {
        item: normalize({
            word: (1 + math.log(count)) * idf[word]
            for word, count in terms.items()
        })
        for item, terms in counts.items()
    }


class Recommender:
    def __init__(self, catalog, purchases):
        self.products = product_vectors(catalog)
        self.histories = defaultdict(Counter)
        self.popularity = Counter()
        for row in purchases:
            user, item = row["user_id"], row["product_id"]
            self.histories[user][item] += 1
            self.popularity[item] += 1
        self.user_vectors = {
            user: normalize(history) for user, history in self.histories.items()
        }
        self.buyers_by_product = defaultdict(list)
        for user, vector in self.user_vectors.items():
            for item, strength in vector.items():
                self.buyers_by_product[item].append((user, strength))

    def signals(self, user):
        seen = self.histories.get(user, {})
        user_vector = self.user_vectors.get(user, {})
        collaborative, profile = Counter(), Counter()
        similarities = defaultdict(float)
        for item, strength in user_vector.items():
            for neighbor, neighbor_strength in self.buyers_by_product[item]:
                if neighbor != user:
                    similarities[neighbor] += strength * neighbor_strength
        for neighbor, similarity in similarities.items():
            for item, strength in self.user_vectors[neighbor].items():
                if item not in seen:
                    collaborative[item] += similarity * strength
        for item, count in seen.items():
            for word, value in self.products[item].items():
                profile[word] += count * value
        profile = normalize(profile)
        # Collaborative scores contain only unseen items with positive evidence.
        scale = max(collaborative.values(), default=1)
        return {
            item: (collaborative[item] / scale, cosine(profile, vector))
            for item, vector in self.products.items() if item not in seen
        }

    def recommend(self, user, limit=10, weight=0.6, popularity_only=False):
        seen = self.histories.get(user, {})
        if not seen or popularity_only:
            ranked = sorted(self.products.keys() - seen.keys(),
                            key=lambda item: (-self.popularity[item], item))
            return [(item, self.popularity[item], "popularity") for item in ranked[:limit]]
        scores = {
            item: weight * collaborative + (1 - weight) * content
            for item, (collaborative, content) in self.signals(user).items()
        }
        ranked = sorted(scores, key=lambda item: (-scores[item], -self.popularity[item], item))
        return [(item, scores[item], "hybrid") for item in ranked[:limit]]


def ndcg(ranked, relevant, limit):
    gains = sum(1 / math.log2(rank + 2) for rank, item in enumerate(ranked)
                if item in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(limit, len(relevant))))
    return gains / ideal


def evaluate(model, purchases, weight, limit, popularity_only=False):
    targets = defaultdict(set)
    for row in purchases:
        targets[row["user_id"]].add(row["product_id"])
    scores = {"warm": [], "cold": []}
    for user, items in targets.items():
        history = model.histories.get(user, {})
        relevant = items - history.keys()
        if relevant:
            group = "warm" if history else "cold"
            ranked = [item for item, _, _ in model.recommend(user, limit, weight, popularity_only)]
            scores[group].append(ndcg(ranked, relevant, limit))
    return {group: (sum(values) / len(values), len(values)) if values else (None, 0)
            for group, values in scores.items()}


def report(label, results):
    parts = []
    for group, (score, count) in results.items():
        value = f"{score:.4f}" if score is not None else "n/a"
        parts.append(f"{group} {value} (n={count})")
    print(f"{label:<19} " + " | ".join(parts))


def benchmark(catalog, purchases, limit):
    dates = sorted({row["date"] for row in purchases})
    if len(dates) < 3:
        raise ValueError("Evaluation needs at least three distinct purchase dates.")
    validation_start = dates[min(len(dates) - 2, max(1, int(len(dates) * 0.6)))]
    test_start = dates[min(len(dates) - 1, max(2, int(len(dates) * 0.8)))]
    train = [row for row in purchases if row["date"] < validation_start]
    validation = [row for row in purchases if validation_start <= row["date"] < test_start]
    test = [row for row in purchases if row["date"] >= test_start]

    print(f"Validation starts {validation_start}. Test starts {test_start}. NDCG@{limit}")
    model = Recommender(catalog, train)
    report("Validation popular", evaluate(model, validation, 0.6, limit, True))
    best_weight, best_score = 0.6, -1
    for weight in (0.6, 0, 0.2, 0.4, 0.5, 0.8, 1):
        scores = evaluate(model, validation, weight, limit)
        report(f"Validation CF {weight:.0%}", scores)
        warm_score = scores["warm"][0]
        if warm_score is not None and warm_score > best_score:
            best_weight, best_score = weight, warm_score

    print(f"Test weight selected from validation: {best_weight:.0%} collaborative")
    model = Recommender(catalog, train + validation)
    report("Test popular", evaluate(model, test, 0.6, limit, True))
    report("Test content only", evaluate(model, test, 0, limit))
    report("Test CF only", evaluate(model, test, 1, limit))
    report("Test 60/40", evaluate(model, test, 0.6, limit))
    report("Test selected", evaluate(model, test, best_weight, limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DATA / "products.csv",
                        help="Product catalog CSV (default: bundled catalog)")
    parser.add_argument("--purchases", type=Path, default=DATA / "purchases.csv",
                        help="Purchase history CSV (default: bundled purchases)")
    commands = parser.add_subparsers(dest="command", required=True)
    recommend = commands.add_parser("recommend", help="Rank unseen products for a shopper")
    recommend.add_argument("user", help="Shopper ID, such as U16 in the sample")
    recommend.add_argument("--weight", type=float, default=0.6,
                           help="Collaborative share from 0 to 1 (default: 0.6)")
    recommend.add_argument("--limit", type=int, default=5,
                           help="Number of recommendations to show (default: 5)")
    evaluate_cmd = commands.add_parser("evaluate", help="Compare rankings on held-out purchases")
    evaluate_cmd.add_argument("--limit", type=int, default=10,
                              help="Ranking depth for NDCG (default: 10)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "recommend" and not 0 <= args.weight <= 1:
        parser.error("--weight must be between 0 and 1")

    try:
        products, catalog, purchases = load_data(args.products, args.purchases)
        if args.command == "evaluate":
            for row in purchases:
                row["date"] = date.fromisoformat(row["date"])
            benchmark(catalog, purchases, args.limit)
            return
    except (OSError, ValueError) as error:
        parser.error(str(error))
    model = Recommender(catalog, purchases)
    for item, score, reason in model.recommend(args.user, args.limit, args.weight):
        print(f"{item:>4}  {products[item]['name']:<24}  {score:.4f}  {reason}")


if __name__ == "__main__":
    main()
