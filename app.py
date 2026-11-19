"""Interactive demo for the bundled shopper histories."""

import streamlit as st

from recommender import Recommender, load_data


def main():
    st.set_page_config(page_title="Shopper recommendations", layout="centered")
    st.title("Shopper recommendations")
    st.caption("Explore recommendations from a sample of synthetic purchases.")

    products, catalog, purchases = load_data()
    model = Recommender(catalog, purchases)
    shoppers = sorted(model.histories)
    choices = [*shoppers, "New shopper"]
    initial = shoppers.index("U16") if "U16" in shoppers else 0
    shopper = st.selectbox("Shopper", choices, index=initial)

    st.subheader("Purchase history")
    history = sorted(
        (row for row in purchases if row["user_id"] == shopper),
        key=lambda row: row["date"],
        reverse=True,
    )
    if history:
        st.dataframe(
            [{"Product": products[row["product_id"]]["name"], "Date": row["date"]} for row in history],
            hide_index=True,
        )
    else:
        st.write("This shopper has no purchases yet.")

    st.subheader("Top recommendations")
    if history:
        st.caption("Ranked by similar shoppers’ purchases (60%) and product text (40%).")
    else:
        st.caption("Ranked by total purchases across all shoppers.")

    ranked = model.recommend(shopper, limit=3)
    if not ranked:
        st.write("There are no more products to recommend.")
        return
    cards = st.columns(len(ranked), border=True)
    for rank, (item, score, method) in enumerate(ranked, start=1):
        product = products[item]
        with cards[rank - 1]:
            st.caption(product["category"])
            st.markdown(f"**{rank}. {product['name']}**")
            if method == "popularity":
                st.caption(f"Purchases: {int(score)}")
            else:
                st.caption(f"Hybrid score {score:.3f}")


if __name__ == "__main__":
    main()
