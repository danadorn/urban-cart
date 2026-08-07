import numpy as np
import pandas as pd

np.random.seed(42)
OUT_DIR = "data/processed"

customers = pd.read_csv(f"{OUT_DIR}/clean_customers.csv")
orders = pd.read_csv(f"{OUT_DIR}/clean_orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(f"{OUT_DIR}/clean_order_items.csv")
products = pd.read_csv(f"{OUT_DIR}/clean_products.csv")

oi = order_items.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id")
oi_pos = oi[oi["quantity"] > 0].copy()  # revenue-generating lines only for RFM/Monetary


# 1. RFM SEGMENTATION (raw NumPy, no pandas.qcut)
print("=" * 70)
print("1. RFM SEGMENTATION")
print("=" * 70)

snapshot_date = orders["order_date"].max()

cust_ids = customers["customer_id"].to_numpy()

# Recency: days since last order (per customer), as a NumPy array
last_order = oi_pos.groupby("customer_id")["order_date"].max()
last_order = last_order.reindex(cust_ids)
recency_days = (snapshot_date - last_order).dt.days.to_numpy(dtype=float)
recency_days = np.where(np.isnan(recency_days), np.nanmax(recency_days) + 1, recency_days)  # never-purchased -> worst

# Frequency: number of distinct orders per customer
freq = oi_pos.groupby("customer_id")["order_id"].nunique()
freq = freq.reindex(cust_ids).to_numpy(dtype=float)
freq = np.nan_to_num(freq, nan=0.0)

# Monetary: total net spend per customer
oi_pos["line_amount"] = oi_pos["quantity"] * oi_pos["unit_price"] * (1 - oi_pos["discount"])
mon = oi_pos.groupby("customer_id")["line_amount"].sum()
mon = mon.reindex(cust_ids).to_numpy(dtype=float)
mon = np.nan_to_num(mon, nan=0.0)


def quintile_score_numpy(arr, higher_is_better=True):
    """
    Bucket a 1D array into 5 quintile-based scores (1-5), computed with
    raw NumPy percentile boundaries -- no pandas.qcut.
    Formula: for each value x, score = which of the 5 bins bounded by the
    20/40/60/80th percentiles of arr it falls into.
    """
    edges = np.percentile(arr, [20, 40, 60, 80])
    scores = np.digitize(arr, edges, right=True) + 1  # 1..5
    if not higher_is_better:
        scores = 6 - scores  # invert so "1" always means "worst"
    return scores


r_score = quintile_score_numpy(recency_days, higher_is_better=False)  # low recency (recent) = good
f_score = quintile_score_numpy(freq, higher_is_better=True)
m_score = quintile_score_numpy(mon, higher_is_better=True)

# Combined RFM score: simple weighted sum, weights reflect that Monetary
# and Frequency matter more to lifetime value than pure Recency for a
# retailer with a multi-week purchase cycle.
rfm_score = 0.2 * r_score + 0.4 * f_score + 0.4 * m_score

rfm_df = pd.DataFrame({
    "customer_id": cust_ids,
    "recency_days": recency_days,
    "frequency": freq,
    "monetary": mon,
    "r_score": r_score, "f_score": f_score, "m_score": m_score,
    "rfm_score": rfm_score,
})


def segment_label(score):
    if score >= 4.2: return "Champions"
    if score >= 3.4: return "Loyal"
    if score >= 2.6: return "Potential"
    if score >= 1.8: return "At Risk"
    return "Churned"

rfm_df["segment"] = rfm_df["rfm_score"].apply(segment_label)
rfm_df.to_csv(f"{OUT_DIR}/rfm_segments.csv", index=False)

print(rfm_df["segment"].value_counts())
print("\nSanity check: manual percentile bucketing vs pandas.qcut (frequency only)")
qcut_check = pd.qcut(freq, 5, labels=False, duplicates="drop")
print("NumPy f_score distribution:", np.unique(f_score, return_counts=True))
print("pandas.qcut distribution:  ", np.unique(qcut_check, return_counts=True))
print("(bucket counts should be broadly similar; used ONLY to verify, not to compute the final score)")


# 2. COSINE SIMILARITY -- product recommendation (raw NumPy)
print("\n" + "=" * 70)
print("2. PRODUCT SIMILARITY / RECOMMENDATION (cosine similarity)")
print("=" * 70)

# Build a customer x product quantity matrix
top_products = oi_pos["product_id"].value_counts().head(80).index  # cap size for a clean dense matrix
mat_df = oi_pos[oi_pos["product_id"].isin(top_products)]
cust_product = mat_df.pivot_table(index="customer_id", columns="product_id",
                                   values="quantity", aggfunc="sum", fill_value=0)
M = cust_product.to_numpy(dtype=float)  # rows=customers, cols=products

def cosine_similarity_matrix(X):
    """
    Cosine similarity between columns of X (here: products), computed
    from raw dot products and norms. Formula:
        sim(i,j) = (x_i . x_j) / (||x_i|| * ||x_j||)
    Implemented as a single matrix multiply for efficiency:
        S = (Xn^T Xn), where Xn has L2-normalized columns.
    """
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1e-9  # avoid divide-by-zero for all-zero columns
    X_norm = X / norms
    return X_norm.T @ X_norm

product_sim = cosine_similarity_matrix(M)
product_ids_order = cust_product.columns.to_numpy()

# Recommend 3 products for 5 sample customers: for each customer, score
# every product by the weighted-average similarity to products they've
# already bought (weighted by how much they bought), excluding owned items.
sample_customers = cust_product.index.to_numpy()[:5]
print("\nSample recommendations (customer_id: recommended product_ids):")
recs = {}
for cid in sample_customers:
    row = cust_product.loc[cid].to_numpy(dtype=float)
    owned = row > 0
    if owned.sum() == 0:
        continue
    scores = product_sim @ row  # weighted similarity to purchase history
    scores[owned] = -np.inf  # never recommend something already bought
    top3_idx = np.argsort(scores)[-3:][::-1]
    recs[cid] = product_ids_order[top3_idx].tolist()
    print(f"  customer {cid}: {recs[cid]}")

print("\nSanity check: comparing one pair's manual cosine similarity vs numpy.dot formula directly")
i, j = 0, 1
a, b = M[:, i], M[:, j]
manual = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
print(f"  manual dot-product cosine sim (products {product_ids_order[i]},{product_ids_order[j]}): {manual:.4f}")
print(f"  matrix-computed value at [0,1]: {product_sim[0,1]:.4f}  (should match)")


# 3. REGRESSION VIA NORMAL EQUATION
print("\n" + "=" * 70)
print("3. REGRESSION VIA NORMAL EQUATION (monthly revenue ~ month index)")
print("=" * 70)

oi_pos["year_month"] = oi_pos["order_date"].dt.to_period("M")
monthly_rev = oi_pos.groupby("year_month")["line_amount"].sum().sort_index()
y = monthly_rev.to_numpy(dtype=float)
x = np.arange(len(y), dtype=float)  # month index 0..N-1

# Design matrix with intercept column
X = np.column_stack([np.ones_like(x), x])

# Normal equation: beta = (X^T X)^-1 X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
beta = XtX_inv @ X.T @ y
intercept, slope = beta

y_pred = X @ beta
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"Model: revenue = {intercept:,.2f} + {slope:,.2f} * month_index")
print(f"R^2 (manual: 1 - SS_res/SS_tot) = {r_squared:.4f}")

# Forecast next 2 months
future_x = np.array([[1, len(y)], [1, len(y) + 1]], dtype=float)
forecast = future_x @ beta
resid_std = np.sqrt(ss_res / (len(y) - 2))
print(f"Forecast next 2 months: {forecast.round(2).tolist()}  "
      f"(+/- ~{1.96*resid_std:,.2f} at ~95% CI using residual std)")

print("\nSanity check vs numpy.polyfit (degree 1, used only to verify):")
slope_pf, intercept_pf = np.polyfit(x, y, 1)
print(f"  polyfit: intercept={intercept_pf:,.2f}, slope={slope_pf:,.2f}  (should match normal-equation result)")


# 4. MONTE CARLO SIMULATION -- stockout probability
print("\n" + "=" * 70)
print("4. MONTE CARLO SIMULATION (stockout risk, 3 products)")
print("=" * 70)

N_TRIALS = 10000
LEAD_TIME_DAYS = 14  # assumed supplier lead time

# pick 3 products with enough order history to estimate a daily demand rate
prod_daily = (oi_pos.groupby(["product_id", oi_pos["order_date"].dt.date])["quantity"]
              .sum().reset_index())
demand_stats = prod_daily.groupby("product_id")["quantity"].agg(["mean", "std", "count"])
candidates = demand_stats[demand_stats["count"] >= 40].copy()
stock_lookup = products.set_index("product_id")["stock_units"]
candidates["stock_units"] = candidates.index.map(stock_lookup)
candidates["stock_to_demand_ratio"] = candidates["stock_units"] / (candidates["mean"] * LEAD_TIME_DAYS)
# choose the 3 products with the TIGHTEST stock-to-demand ratio -- these
# are the genuine stockout-risk candidates leadership should worry about
chosen = candidates.dropna(subset=["stock_units"]).sort_values("stock_to_demand_ratio").head(3)

results = []
for pid, row in chosen.iterrows():
    mu, sigma = max(row["mean"], 0.01), max(row["std"], 0.01)
    # simulate daily demand over the lead-time window, N_TRIALS times,
    # using a Poisson-like Gaussian approximation clipped at 0
    sim_demand = np.random.normal(loc=mu, scale=sigma, size=(N_TRIALS, LEAD_TIME_DAYS))
    sim_demand = np.clip(sim_demand, 0, None)
    total_demand_per_trial = sim_demand.sum(axis=1)

    current_stock = products.loc[products["product_id"] == pid, "stock_units"]
    current_stock = float(current_stock.iloc[0]) if len(current_stock) and pd.notna(current_stock.iloc[0]) else mu * LEAD_TIME_DAYS

    stockout_flags = total_demand_per_trial > current_stock
    p_stockout = stockout_flags.mean()
    se = np.sqrt(p_stockout * (1 - p_stockout) / N_TRIALS)
    ci_low, ci_high = p_stockout - 1.96 * se, p_stockout + 1.96 * se

    reorder_point = np.percentile(total_demand_per_trial, 95)  # 95th-percentile demand as safety stock target

    results.append({
        "product_id": pid, "daily_mean_demand": round(mu, 2),
        "current_stock": round(current_stock, 1),
        "p_stockout": round(p_stockout, 4),
        "ci_95_low": round(max(ci_low, 0), 4), "ci_95_high": round(min(ci_high, 1), 4),
        "recommended_reorder_point": round(reorder_point, 1),
    })
    print(f"Product {pid}: P(stockout in {LEAD_TIME_DAYS}d) = {p_stockout:.2%} "
          f"(95% CI [{max(ci_low,0):.2%}, {min(ci_high,1):.2%}]), "
          f"recommended reorder point = {reorder_point:.0f} units")

mc_df = pd.DataFrame(results)
mc_df.to_csv(f"{OUT_DIR}/monte_carlo_stockout.csv", index=False)

print(f"\nSanity check: with {N_TRIALS} trials, standard error on a p~0.1 estimate is "
      f"~{np.sqrt(0.1*0.9/N_TRIALS):.4f}, giving a tight, stable 95% CI -- consistent with above.")

rfm_df.to_csv(f"{OUT_DIR}/rfm_segments.csv", index=False)
print("\n[DONE] Phase 3 outputs saved to data/processed/")
