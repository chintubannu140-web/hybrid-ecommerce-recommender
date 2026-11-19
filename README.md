# Hybrid E-Commerce Recommendation System

A Python recommender that suggests products a shopper hasn’t bought yet. It combines purchase patterns from similar shoppers with TF-IDF similarity between product descriptions.

## Quick start

Run these commands from the project directory with Python 3.10 or later. The demo uses the standard library and the included CSV files.

```bash
python3 recommender.py recommend U16 --limit 3
python3 recommender.py recommend new-shopper --limit 3
```

`U16` bought trail running shoes. Their top three recommendations are a sports water bottle, running socks, and a running belt. For `new-shopper`, the most frequently purchased item comes first.

## Web demo

From the project directory, install Streamlit in a virtual environment and start the app.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Choose a shopper to see their purchase history and top three recommendations, or select **New shopper** to see the popularity fallback.

## How ranking works

- **Collaborative filtering** finds shoppers who bought at least one of the same products, using an index of buyers by product. Shoppers with more similar purchase histories contribute more to each candidate’s score.
- **Content-based filtering** applies TF-IDF to product names, categories, and descriptions. A shopper’s purchases form a weighted text profile. Cosine similarity measures how closely each candidate matches it.
- **Final ranking** combines 60% collaborative score and 40% content score by default. Use `--weight` to change the collaborative share. Shoppers with no history get products ranked by purchase count.

## Evaluation

```bash
python3 recommender.py evaluate
```

The benchmark splits purchases by date, chooses a blend on validation data, then tests on later purchases. NDCG@10 rewards rankings that place a shopper’s newly purchased items near the top.

| Method | Test NDCG@10 · 15 returning shoppers |
| --- | ---: |
| Popularity | 0.3690 |
| Content only | 0.8812 |
| Collaborative only | 0.8821 |
| Hybrid (60/40) | 0.8762 |

The sample is synthetic, with 12 products and 64 purchases. Several blends tied on validation. Collaborative filtering alone scored highest on the test set. The two new shoppers received the same popularity ranking under every method and averaged 0.5089.

## Use your own data

Match the headers in [products.csv](data/products.csv) and [purchases.csv](data/purchases.csv). Each purchase row represents one interaction and must reference a product in the catalog. For evaluation, include at least three distinct dates in `YYYY-MM-DD` format.

```bash
python3 recommender.py --products products.csv --purchases purchases.csv evaluate
```
