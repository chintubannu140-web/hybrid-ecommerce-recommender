# Hybrid E-Commerce Recommendation System

A Python recommender that suggests products a shopper hasn’t bought yet. It fits TF-IDF to **purchase histories** for collaborative filtering and to **product metadata** for content-based filtering.

## How ranking works

1. **Find similar shoppers.** Each shopper’s purchase history is a document, and each product ID is a term. Repeat purchases increase the product’s weight. Products bought by many shoppers get a lower IDF, so sharing an unusual purchase is a stronger signal. A product-to-buyers index limits cosine comparisons to shoppers with overlapping purchases.
2. **Find similar products.** Each product’s name, category, and description form a text document. Purchased products contribute their TF-IDF vectors to a shopper profile. Cosine similarity scores each unseen product against that profile.
3. **Rank unseen products.** Scale the collaborative score by the highest candidate score, then compute `0.60 × collaborative + 0.40 × content`. For a shopper with no history, rank the catalog by total purchase count.

## Quick start

Run these commands from the project directory with Python 3.10 or later. The demo uses the standard library and the included CSV files.

```bash
python3 recommender.py recommend U16 --limit 3
python3 recommender.py recommend new-shopper --limit 3
```

`U16` bought trail running shoes. The index finds the 5 other buyers of those shoes among 18 other shoppers. The sports water bottle ranks first with a collaborative score of 1.000 and a content score of 0.234, giving a hybrid score of 0.694. For `new-shopper`, the pour-over brewer ranks first with 7 purchases.

## Optional web demo

From the project directory, install Streamlit in a virtual environment and start the app.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Choose a shopper to see their purchase history and top three recommendations, or select **New shopper** to see the popularity fallback.

## Evaluation

```bash
python3 recommender.py evaluate
```

The benchmark splits purchases by date, chooses a blend on validation data, then tests on later purchases. NDCG@10 rewards rankings that place a shopper’s newly purchased items near the top.

| Method | Test NDCG@10 · 15 returning shoppers |
| --- | ---: |
| Popularity | 0.3690 |
| Content only | 0.8812 |
| Collaborative only | 0.8775 |
| Hybrid (60/40) | 0.8762 |

The sample is synthetic, with 12 products and 64 purchases. Several blends tied on validation, including the 60/40 default. Content-based filtering alone scored highest on the test set. The two new shoppers received the same popularity ranking under every method and averaged 0.5089.

## Use your own data

Match the headers in [products.csv](data/products.csv) and [purchases.csv](data/purchases.csv). Each purchase row represents one interaction and must reference a product in the catalog. For evaluation, include at least three distinct dates in `YYYY-MM-DD` format.

```bash
python3 recommender.py --products products.csv --purchases purchases.csv evaluate
```
