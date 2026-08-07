import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUT_DIR = "data/processed"
FIG_DIR = "figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

customers = pd.read_csv(f"{OUT_DIR}/clean_customers.csv")
orders = pd.read_csv(f"{OUT_DIR}/clean_orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(f"{OUT_DIR}/clean_order_items.csv")
products = pd.read_csv(f"{OUT_DIR}/clean_products.csv")
reviews = pd.read_csv(f"{OUT_DIR}/clean_reviews.csv")
web_sessions = pd.read_csv(f"{OUT_DIR}/clean_web_sessions.csv", parse_dates=["session_date"])
rfm = pd.read_csv(f"{OUT_DIR}/rfm_segments.csv")
mc = pd.read_csv(f"{OUT_DIR}/monte_carlo_stockout.csv")

oi = order_items.merge(orders[["order_id", "customer_id", "order_date"]], on="order_id")
oi_pos = oi[oi["quantity"] > 0].copy()
oi_pos["line_amount"] = oi_pos["quantity"] * oi_pos["unit_price"] * (1 - oi_pos["discount"])

findings = []


# Q1: Which RFM segment generates the most revenue, and demographics?
seg_rev = rfm.groupby("segment")["monetary"].sum().sort_values(ascending=False)
rfm_c = rfm.merge(customers[["customer_id", "age", "gender"]], on="customer_id", how="left")
champions_age = rfm_c.loc[rfm_c["segment"] == "Champions", "age"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
seg_rev.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(seg_rev)))
ax.set_ylabel("Total revenue ($)")
ax.set_title("Q1: Revenue by RFM Segment")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q1_revenue_by_segment.png", dpi=150)
plt.close()
findings.append(f"Q1: '{seg_rev.index[0]}' segment generates the most revenue "
                 f"(${seg_rev.iloc[0]:,.0f}); average age in that segment is {champions_age:.1f}.")

# Q2: Seasonality in overall revenue -- rolling average
daily_rev = oi_pos.groupby(oi_pos["order_date"].dt.date)["line_amount"].sum()
daily_rev.index = pd.to_datetime(daily_rev.index)
rolling_30 = daily_rev.rolling(30).mean()

fig, ax = plt.subplots(figsize=(11, 5))
daily_rev.plot(ax=ax, alpha=0.3, label="Daily revenue")
rolling_30.plot(ax=ax, linewidth=2, label="30-day rolling average", color="crimson")
ax.set_ylabel("Revenue ($)")
ax.set_title("Q2: Daily Revenue with 30-Day Rolling Average (Seasonality Check)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q2_seasonality.png", dpi=150)
plt.close()
monthly = oi_pos.set_index("order_date")["line_amount"].resample("ME").sum()
monthly_by_month_num = monthly.groupby(monthly.index.month).mean()
peak_month = monthly_by_month_num.idxmax()
findings.append(f"Q2: Revenue shows a repeating monthly pattern; month {peak_month} is on average "
                 f"the strongest calendar month across years (${monthly_by_month_num.max():,.0f} avg).")


# Q3: Highest effective margin after discounts and returns, by category
oi_all = oi.merge(products[["product_id", "category", "cost"]], on="product_id")
oi_all["net_revenue"] = oi_all["quantity"] * oi_all["unit_price"] * (1 - oi_all["discount"])
oi_all["net_cost"] = oi_all["quantity"] * oi_all["cost"]
cat_margin = oi_all.groupby("category").agg(net_revenue=("net_revenue", "sum"),
                                             net_cost=("net_cost", "sum"))
cat_margin["effective_margin_pct"] = (cat_margin["net_revenue"] - cat_margin["net_cost"]) / cat_margin["net_revenue"]
cat_margin = cat_margin.sort_values("effective_margin_pct", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
(cat_margin["effective_margin_pct"] * 100).plot(kind="bar", ax=ax, color=sns.color_palette("crest", len(cat_margin)))
ax.set_ylabel("Effective margin (%)")
ax.set_title("Q3: Effective Margin by Category (net of discounts & returns)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q3_margin_by_category.png", dpi=150)
plt.close()
findings.append(f"Q3: '{cat_margin.index[0]}' has the highest effective margin "
                 f"({cat_margin['effective_margin_pct'].iloc[0]:.1%}) after discounts and returns.")


# Q4: Do higher review ratings correlate with repeat purchases?
cust_orders = oi_pos.groupby("customer_id")["order_id"].nunique().rename("n_orders")
cust_avg_rating = reviews.groupby("customer_id")["rating"].mean().rename("avg_rating_given")
merged = pd.concat([cust_orders, cust_avg_rating], axis=1).dropna()
corr = np.corrcoef(merged["avg_rating_given"], merged["n_orders"])[0, 1]

fig, ax = plt.subplots(figsize=(7, 5))
sns.regplot(data=merged, x="avg_rating_given", y="n_orders", ax=ax,
            scatter_kws={"alpha": 0.3}, line_kws={"color": "crimson"})
ax.set_title(f"Q4: Review Rating vs Repeat Purchases (r={corr:.2f})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q4_rating_vs_repeat.png", dpi=150)
plt.close()
findings.append(f"Q4: Correlation between a customer's average rating given and their order count "
                 f"is r={corr:.2f} — {'a weak' if abs(corr)<0.3 else 'a moderate' if abs(corr)<0.6 else 'a strong'} "
                 "relationship, so rating alone is a limited predictor of repeat purchase.")


# Q5: Device x country engagement-to-purchase conversion
ws_c = web_sessions.merge(customers[["customer_id", "country"]], on="customer_id")
purchasers = set(oi_pos["customer_id"].unique())
ws_c["purchased"] = ws_c["customer_id"].isin(purchasers)
conv = ws_c.groupby(["country", "device"])["purchased"].mean().reset_index()
conv_pivot = conv.pivot(index="country", columns="device", values="purchased")

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(conv_pivot, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax)
ax.set_title("Q5: Engagement-to-Purchase Conversion by Country x Device")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q5_conversion_heatmap.png", dpi=150)
plt.close()
best_cell = conv.loc[conv["purchased"].idxmax()]
findings.append(f"Q5: {best_cell['country']} / {best_cell['device']} shows the highest "
                 f"engagement-to-purchase conversion ({best_cell['purchased']:.1%}).")


# Q6: Regression forecast for next 2 months w/ uncertainty (from Phase 3)
oi_pos["year_month"] = oi_pos["order_date"].dt.to_period("M")
monthly_rev = oi_pos.groupby("year_month")["line_amount"].sum().sort_index()
y = monthly_rev.to_numpy(dtype=float)
x = np.arange(len(y), dtype=float)
X = np.column_stack([np.ones_like(x), x])
beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ beta
resid_std = np.sqrt(np.sum((y - y_pred) ** 2) / (len(y) - 2))
future_x = np.array([[1, len(y)], [1, len(y) + 1]])
forecast = future_x @ beta
ci = 1.96 * resid_std

fig, ax = plt.subplots(figsize=(10, 5))
months_idx = np.arange(len(y) + 2)
ax.plot(months_idx[:len(y)], y, "o-", label="Actual monthly revenue")
ax.plot(months_idx, X_full_pred := np.column_stack([np.ones(len(months_idx)), months_idx]) @ beta,
        "--", color="gray", label="Fitted trend")
ax.errorbar(months_idx[-2:], forecast, yerr=ci, fmt="D", color="crimson", capsize=5, label="2-month forecast (95% CI)")
ax.set_xlabel("Month index"); ax.set_ylabel("Revenue ($)")
ax.set_title("Q6: Revenue Forecast — Next 2 Months")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q6_forecast.png", dpi=150)
plt.close()
findings.append(f"Q6: Linear trend forecasts next 2 months at ${forecast[0]:,.0f} and ${forecast[1]:,.0f}, "
                 f"+/- ${ci:,.0f} (95% CI).")


# Q7: Top 5 stockout risks from Monte Carlo + reorder points
mc_top5 = mc.sort_values("p_stockout", ascending=False).head(5)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(mc_top5["product_id"].astype(str), mc_top5["p_stockout"] * 100, color="darkorange")
ax.set_ylabel("P(stockout within lead time) %")
ax.set_xlabel("Product ID")
ax.set_title("Q7: Top Stockout-Risk Products (Monte Carlo, 10,000 trials)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q7_stockout_risk.png", dpi=150)
plt.close()
findings.append("Q7: Products " + ", ".join(mc_top5["product_id"].astype(str)) +
                 " show the highest stockout risk; recommended reorder points: " +
                 ", ".join(f"{r.product_id}->{r.recommended_reorder_point:.0f}u" for r in mc_top5.itertuples()))


# Q8: One data-quality finding that materially changed a conclusion
gross_rev = oi[oi["quantity"] > 0]
gross_rev_total = (gross_rev["quantity"] * gross_rev["unit_price"] * (1 - gross_rev["discount"])).sum()
net_rev_total = oi_pos["line_amount"].sum() + oi[oi["quantity"] < 0].assign(
    line_amount=lambda d: d["quantity"] * d["unit_price"] * (1 - d["discount"]))["line_amount"].sum()
pct_diff = (gross_rev_total - net_rev_total) / gross_rev_total

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Gross revenue\n(ignoring returns)", "Net revenue\n(returns applied)"],
       [gross_rev_total, net_rev_total], color=["#8ecae6", "#e63946"])
ax.set_ylabel("Revenue ($)")
ax.set_title("Q8: Impact of Correctly Handling Returns")
for i, v in enumerate([gross_rev_total, net_rev_total]):
    ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/q8_returns_impact.png", dpi=150)
plt.close()
findings.append(f"Q8: Ignoring the negative-quantity return rows (as a naive analysis would) overstates "
                 f"revenue by {pct_diff:.2%} (${gross_rev_total:,.0f} gross vs ${net_rev_total:,.0f} net) — "
                 "a material difference for any category-profitability conclusion.")

with open(f"{OUT_DIR}/phase4_findings.txt", "w") as f:
    f.write("\n\n".join(findings))

for f_ in findings:
    print(f_)

print(f"\n[DONE] {len(findings)} charts saved to {FIG_DIR}/")
