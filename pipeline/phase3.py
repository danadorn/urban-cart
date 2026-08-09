import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT_DIR = Path("data/processed")


def quintile_score_numpy(arr, higher_is_better=True):
    edges = np.percentile(arr, [20, 40, 60, 80])
    scores = np.digitize(arr, edges, right=True) + 1
    if not higher_is_better:
        scores = 6 - scores
    return scores


def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1e-9
    X_norm = X / norms
    return X_norm.T @ X_norm


def run_phase3() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(OUT_DIR / "clean_customers.csv")
    orders = pd.read_csv(OUT_DIR / "clean_orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(OUT_DIR / "clean_order_items.csv")
    products = pd.read_csv(OUT_DIR / "clean_products.csv")

    oi = order_items.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id")
    oi_pos = oi[oi["quantity"] > 0].copy()
    oi_pos["line_amount"] = oi_pos["quantity"] * oi_pos["unit_price"] * (1 - oi_pos["discount"])

    print("=" * 70)
    print("1. RFM SEGMENTATION")
    print("=" * 70)

    snapshot_date = orders["order_date"].max()
    cust_ids = customers["customer_id"].to_numpy()

    last_order = oi_pos.groupby("customer_id")["order_date"].max()
    last_order = last_order.reindex(cust_ids)
    recency_days = (snapshot_date - last_order).dt.days.to_numpy(dtype=float)
    recency_days = np.where(np.isnan(recency_days), np.nanmax(recency_days) + 1, recency_days)

    freq = oi_pos.groupby("customer_id")["order_id"].nunique()
    freq = freq.reindex(cust_ids).to_numpy(dtype=float)
    freq = np.nan_to_num(freq, nan=0.0)

    mon = oi_pos.groupby("customer_id")["line_amount"].sum()
    mon = mon.reindex(cust_ids).to_numpy(dtype=float)
    mon = np.nan_to_num(mon, nan=0.0)

    r_score = quintile_score_numpy(recency_days, higher_is_better=False)
    f_score = quintile_score_numpy(freq, higher_is_better=True)
    m_score = quintile_score_numpy(mon, higher_is_better=True)

    rfm_score = 0.2 * r_score + 0.4 * f_score + 0.4 * m_score

    rfm_df = pd.DataFrame({
        "customer_id": cust_ids,
        "recency_days": recency_days,
        "frequency": freq,
        "monetary": mon,
        "r_score": r_score,
        "f_score": f_score,
        "m_score": m_score,
        "rfm_score": rfm_score,
    })

    def segment_label(score):
        if score >= 4.2:
            return "Champions"
        if score >= 3.4:
            return "Loyal"
        if score >= 2.6:
            return "Potential"
        if score >= 1.8:
            return "At Risk"
        return "Churned"

    rfm_df["segment"] = rfm_df["rfm_score"].apply(segment_label)
    rfm_df.to_csv(OUT_DIR / "rfm_segments.csv", index=False)

    print(rfm_df["segment"].value_counts())
    print("\nSanity check: manual percentile bucketing vs pandas.qcut (frequency only)")
    qcut_check = pd.qcut(freq, 5, labels=False, duplicates="drop")
    print("NumPy f_score distribution:", np.unique(f_score, return_counts=True))
    print("pandas.qcut distribution:  ", np.unique(qcut_check, return_counts=True))

    print("\n" + "=" * 70)
    print("2. PRODUCT SIMILARITY / RECOMMENDATION (cosine similarity)")
    print("=" * 70)

    TOP_K = 80
    top_products = oi_pos["product_id"].value_counts().head(TOP_K).index
    mat_df = oi_pos[oi_pos["product_id"].isin(top_products)]
    cust_product = mat_df.pivot_table(
        index="customer_id",
        columns="product_id",
        values="quantity",
        aggfunc="sum",
        fill_value=0,
    )
    M = cust_product.to_numpy(dtype=float)

    product_sim = cosine_similarity_matrix(M)
    product_ids_order = cust_product.columns.to_numpy()

    sample_customers = cust_product.index.to_numpy()[:5]
    print("\nSample recommendations (customer_id: recommended product_ids):")
    for cid in sample_customers:
        row = cust_product.loc[cid].to_numpy(dtype=float)
        owned = row > 0
        if owned.sum() == 0:
            continue
        scores = product_sim @ row
        scores[owned] = -np.inf
        top3_idx = np.argsort(scores)[-3:][::-1]
        print(f"  customer {cid}: {product_ids_order[top3_idx].tolist()}")

    print("\nSanity check: comparing one pair's manual cosine similarity vs matrix result")
    i, j = 0, 1
    if M.shape[1] > 1:
        a, b = M[:, i], M[:, j]
        manual = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        print(f"  manual cosine sim = {manual:.4f}, matrix value = {product_sim[i, j]:.4f}")

    print("\n" + "=" * 70)
    print("3. REGRESSION VIA NORMAL EQUATION (monthly revenue ~ month index)")
    print("=" * 70)

    oi_pos["year_month"] = oi_pos["order_date"].dt.to_period("M")
    monthly_rev = oi_pos.groupby("year_month")["line_amount"].sum().sort_index()
    y = monthly_rev.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    XtX = X.T @ X
    beta = np.linalg.inv(XtX) @ X.T @ y
    intercept, slope = beta

    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    print(f"Model: revenue = {intercept:,.2f} + {slope:,.2f} * month_index")
    print(f"R^2 = {r_squared:.4f}")

    future_x = np.array([[1, len(y)], [1, len(y) + 1]], dtype=float)
    forecast = future_x @ beta
    resid_std = np.sqrt(ss_res / max(len(y) - 2, 1))
    print(f"Forecast next 2 months: {forecast.round(2).tolist()} (+/- {1.96 * resid_std:.2f})")

    print("\n" + "=" * 70)
    print("4. MONTE CARLO SIMULATION (stockout risk, 3 products)")
    print("=" * 70)

    N_TRIALS = 10000
    LEAD_TIME_DAYS = 14
    prod_daily = (
        oi_pos.groupby(["product_id", oi_pos["order_date"].dt.date])["quantity"]
        .sum()
        .reset_index()
    )
    demand_stats = prod_daily.groupby("product_id")["quantity"].agg(["mean", "std", "count"])
    candidates = demand_stats[demand_stats["count"] >= 40].copy()
    stock_lookup = products.set_index("product_id")["stock_units"]
    candidates["stock_units"] = candidates.index.map(stock_lookup)
    candidates = candidates.dropna(subset=["stock_units"])
    candidates["stock_units"] = candidates["stock_units"].astype(float)

    def stockout_prob(mean, std, stock, lead_time, trials):
        draws = np.random.normal(mean, std, size=(trials, lead_time))
        draws = np.clip(draws, 0, None)
        demand = draws.sum(axis=1)
        return np.mean(demand > stock)

    records = []
    for product_id, row in candidates.iterrows():
        p_stockout = stockout_prob(
            row["mean"], row["std"], row["stock_units"], LEAD_TIME_DAYS, N_TRIALS
        )
        reorder_point = max(0, int(np.ceil(row["mean"] * LEAD_TIME_DAYS + 2 * row["std"] * np.sqrt(LEAD_TIME_DAYS))))
        records.append(
            {
                "product_id": product_id,
                "mean_daily_demand": row["mean"],
                "std_daily_demand": row["std"],
                "stock_units": row["stock_units"],
                "p_stockout": p_stockout,
                "recommended_reorder_point": reorder_point,
            }
        )

    mc_df = pd.DataFrame(records)
    mc_df.to_csv(OUT_DIR / "monte_carlo_stockout.csv", index=False)

    print(f"\n[DONE] All processed files written to {OUT_DIR}/")


if __name__ == "__main__":
    run_phase3()
