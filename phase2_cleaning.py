
import re
import sqlite3
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

DB_PATH = "ecommerce.db"
LEGACY_CSV = "legacy_customers_export.csv"
CATALOG_CSV = "product_catalog_2024.csv"
OUT_DIR = "data/processed"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)


# 1. LOAD RAW SOURCES
con = sqlite3.connect(DB_PATH)
customers = pd.read_sql("SELECT * FROM customers", con)
products = pd.read_sql("SELECT * FROM products", con)
orders = pd.read_sql("SELECT * FROM orders", con, parse_dates=["order_date"])
order_items = pd.read_sql("SELECT * FROM order_items", con)
reviews = pd.read_sql("SELECT * FROM reviews", con, parse_dates=["review_date"])
web_sessions = pd.read_sql("SELECT * FROM web_sessions", con, parse_dates=["session_date"])
con.close()

log(f"[LOAD] customers={customers.shape}, products={products.shape}, orders={orders.shape}, "
    f"order_items={order_items.shape}, reviews={reviews.shape}, web_sessions={web_sessions.shape}")


# 1b. LOAD 4 OF THE PHASE 1 SQL QUERIES DIRECTLY INTO PANDAS
con = sqlite3.connect(DB_PATH)

# Query 1 — revenue/orders/AOV by category (net of discounts)
sql_q1_category_revenue = pd.read_sql("""
    SELECT
        p.category,
        COUNT(DISTINCT o.order_id) AS order_count,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS total_revenue,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount))
              / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
    FROM order_items oi
    JOIN products p ON p.product_id = oi.product_id
    JOIN orders o   ON o.order_id   = oi.order_id
    WHERE oi.quantity > 0
    GROUP BY p.category
    ORDER BY total_revenue DESC
""", con)

# Query 2 — top 20 customers by lifetime spend
sql_q2_top20_customers = pd.read_sql("""
    SELECT
        c.customer_id, c.name, c.city, c.signup_date,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS lifetime_spend
    FROM customers c
    JOIN orders o      ON o.customer_id = c.customer_id
    JOIN order_items oi ON oi.order_id  = o.order_id
    WHERE oi.quantity > 0
    GROUP BY c.customer_id, c.name, c.city, c.signup_date
    ORDER BY lifetime_spend DESC
    LIMIT 20
""", con)

# Query 4 — return rate by category
sql_q4_return_rate_by_category = pd.read_sql("""
    WITH category_items AS (
        SELECT p.category, oi.quantity
        FROM order_items oi
        JOIN products p ON p.product_id = oi.product_id
    )
    SELECT
        category,
        COUNT(*) AS total_line_items,
        SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) AS return_line_items,
        ROUND(1.0 * SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS return_rate
    FROM category_items
    GROUP BY category
    ORDER BY return_rate DESC
""", con)

# Query 9 — payment-method mix by country
sql_q9_payment_mix_by_country = pd.read_sql("""
    WITH country_totals AS (
        SELECT c.country, COUNT(*) AS total_orders
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY c.country
    )
    SELECT
        c.country, o.payment_method, COUNT(*) AS order_count,
        ROUND(1.0 * COUNT(*) / ct.total_orders, 4) AS share_of_country_orders
    FROM orders o
    JOIN customers c       ON c.customer_id = o.customer_id
    JOIN country_totals ct ON ct.country    = c.country
    GROUP BY c.country, o.payment_method
    ORDER BY c.country, share_of_country_orders DESC
""", con)
con.close()

log(f"[SQL->pandas] loaded 4 Phase 1 queries directly into DataFrames: "
    f"Q1 category revenue {sql_q1_category_revenue.shape}, "
    f"Q2 top-20 customers {sql_q2_top20_customers.shape}, "
    f"Q4 return rate by category {sql_q4_return_rate_by_category.shape}, "
    f"Q9 payment mix by country {sql_q9_payment_mix_by_country.shape}")

# save these so Phase 3/4 (or the report) can reuse the *exact* SQL-derived
# figures rather than only the independently-recomputed pandas versions
import os as _os
_os.makedirs("data/processed", exist_ok=True)
sql_q1_category_revenue.to_csv("data/processed/sql_q1_category_revenue.csv", index=False)
sql_q2_top20_customers.to_csv("data/processed/sql_q2_top20_customers.csv", index=False)
sql_q4_return_rate_by_category.to_csv("data/processed/sql_q4_return_rate_by_category.csv", index=False)
sql_q9_payment_mix_by_country.to_csv("data/processed/sql_q9_payment_mix_by_country.csv", index=False)


# 2. CLEAN order_items: exact duplicates + negative-quantity returns
before = len(order_items)
dup_cols = ["order_id", "product_id", "quantity", "unit_price", "discount"]
n_dupes = order_items.duplicated(subset=dup_cols, keep="first").sum()
order_items_clean = order_items.drop_duplicates(subset=dup_cols, keep="first").copy()
log(f"[order_items] removed {n_dupes} exact-duplicate rows "
    f"({n_dupes/before:.2%}); {before} -> {len(order_items_clean)}")

# Policy: negative quantity = a return. We KEEP these rows (they are real
# events, not errors) but flag them explicitly and compute two revenue
# figures downstream: gross (positive lines only) and net (positive +
# negative, i.e. after returns). Silently dropping them would overstate
# revenue and hide the true return impact leadership asked about.
order_items_clean["is_return"] = order_items_clean["quantity"] < 0
order_items_clean["line_amount"] = (
    order_items_clean["quantity"]
    * order_items_clean["unit_price"]
    * (1 - order_items_clean["discount"])
)
log(f"[order_items] {order_items_clean['is_return'].sum()} return line-items flagged "
    f"({order_items_clean['is_return'].mean():.2%} of lines)")

# Cross-check against sql_q4_return_rate_by_category (loaded above straight
# from Phase 1's SQL): overall return rate from the SQL query (run on raw,
# pre-dedup order_items) should be close to the pandas-computed rate on the
# cleaned table -- confirms dedup didn't distort the return-rate signal.
_sql_overall_return_rate = (
    sql_q4_return_rate_by_category["return_line_items"].sum()
    / sql_q4_return_rate_by_category["total_line_items"].sum()
)
_pandas_overall_return_rate = order_items_clean["is_return"].mean()
log(f"[cross-check] overall return rate: SQL (raw table) = {_sql_overall_return_rate:.4f}, "
    f"pandas (deduped table) = {_pandas_overall_return_rate:.4f} "
    f"(small gap expected: SQL query ran on raw order_items before dedup)")


# 3. CLEAN reviews: out-of-range ratings, missing text
before = len(reviews)
bad_rating = ~reviews["rating"].between(1, 5)
log(f"[reviews] {bad_rating.sum()} rows with out-of-range rating "
    f"(values found: {sorted(reviews.loc[bad_rating, 'rating'].unique())}) -> dropped. "
    "Policy: a rating outside 1-5 is a data-entry error, not a real observation, "
    "and cannot be reasonably imputed, so these rows are dropped rather than clamped "
    "(clamping -1->1 or 6->5 would fabricate agreement/disagreement that wasn't given).")
reviews_clean = reviews.loc[~bad_rating].copy()

# review_text: ~20% missing. Policy: keep the row (rating is still valid
# signal) but flag missing text explicitly rather than imputing text or
# dropping the row, since rating-based analyses don't need review_text.
reviews_clean["has_text"] = reviews_clean["review_text"].notna()
log(f"[reviews] {before-len(reviews_clean)} rows dropped for bad rating; "
    f"{(~reviews_clean['has_text']).sum()} of remaining {len(reviews_clean)} rows "
    f"have missing review_text ({(~reviews_clean['has_text']).mean():.1%}), flagged not dropped")


# 4. CLEAN products: price outliers via IQR
q1, q3 = products["unit_price"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
is_outlier = ~products["unit_price"].between(lower, upper)
log(f"[products] IQR bounds on unit_price: [{lower:.2f}, {upper:.2f}] "
    f"(Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}); {is_outlier.sum()} outliers flagged "
    f"({is_outlier.mean():.2%}). Policy: 1.5*IQR is the standard Tukey fence and matches "
    "the README's description of these as rare data-entry errors rather than a genuine "
    "luxury tier; flagged rows are excluded from price-sensitive aggregates (e.g. average "
    "price) but retained in the table so order_items referencing them still resolve.")
products_clean = products.copy()
products_clean["price_outlier"] = is_outlier


# 5. CLEAN customers (from DB): missing age/city/gender
customers_clean = customers.copy()
# city: ~3% missing -> flag as 'Unknown', do not drop (would lose valid
# order history for that customer elsewhere in the pipeline)
customers_clean["city"] = customers_clean["city"].fillna("Unknown")
# age: ~6% missing -> flag with a boolean rather than imputing a mean,
# since imputed ages would silently bias any age-based segmentation
customers_clean["age_missing"] = customers_clean["age"].isna()
# gender: keep NaN as explicit category 'Not specified' (do not drop rows;
# gender is not needed for revenue/RFM), do not guess
customers_clean["gender"] = customers_clean["gender"].fillna("Not specified")
log(f"[customers] city missing -> 'Unknown' ({customers['city'].isna().sum()} rows); "
    f"age missing -> flagged, left NaN ({customers['age'].isna().sum()} rows), NOT imputed "
    "to avoid biasing age-based segments; gender missing -> 'Not specified' "
    f"({customers['gender'].isna().sum()} rows)")


# 6. LEGACY CSV: standardize dates, casing, dedupe, junk rows
leg = pd.read_csv(LEGACY_CSV)
leg.columns = [c.strip() for c in leg.columns]
leg = leg.rename(columns={
    "Customer Name": "name_raw",
    "EMAIL_ADDR": "email",
    "Signup_Dt": "signup_date_raw",
    "Home City": "city",
    "Marketing Segment": "marketing_segment",
})
before = len(leg)

# drop fully-blank rows
leg = leg.dropna(how="all")

# drop obvious junk/test rows (case-insensitive match on name or email)
junk_mask = (
    leg["name_raw"].astype(str).str.strip().str.lower().isin(["test account", "test", "n/a", ""])
    | leg["email"].astype(str).str.strip().str.lower().isin(["test@test.com"])
)
n_junk = junk_mask.sum()
leg = leg.loc[~junk_mask].copy()

# normalize name casing: Title Case, trimmed
leg["name"] = leg["name_raw"].astype(str).str.strip().str.title()

# normalize whitespace on all string cols
for c in ["email", "city", "marketing_segment"]:
    leg[c] = leg[c].astype(str).str.strip().replace({"nan": np.nan})

# standardize 4 inconsistent date formats -> single datetime dtype
def parse_legacy_date(val):
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(val, errors="coerce")  # last-resort fallback

leg["signup_date"] = leg["signup_date_raw"].apply(parse_legacy_date)
n_unparsed = leg["signup_date"].isna().sum()

# de-duplicate: exact email dup, and case-insensitive-name + same city as
# a fuzzy fallback for rows missing email
leg_sorted = leg.sort_values("signup_date")
exact_dup = leg_sorted.duplicated(subset=["email"], keep="last") & leg_sorted["email"].notna()
n_exact_dup = exact_dup.sum()
leg_dedup = leg_sorted.loc[~exact_dup].copy()

fuzzy_key = leg_dedup["name"].str.lower().str.strip() + "|" + leg_dedup["city"].str.lower().str.strip()
fuzzy_dup = leg_dedup.duplicated(subset=None, keep="last") | leg_dedup.assign(_k=fuzzy_key).duplicated(subset="_k", keep="last")
n_fuzzy_dup = fuzzy_dup.sum()
leg_dedup = leg_dedup.loc[~fuzzy_dup].copy()

leg_clean = leg_dedup[["name", "email", "signup_date", "city", "marketing_segment"]].reset_index(drop=True)

log(f"[legacy_customers] {before} raw rows -> dropped {n_junk} junk/test rows, "
    f"{leg['email'].isna().sum()} missing emails retained (flagged via NaN), "
    f"{n_unparsed} dates failed to parse across 4 formats (YYYY-MM-DD, DD-Mon-YYYY, "
    f"Month DD, YYYY, MM/DD/YYYY), removed {n_exact_dup} exact email duplicates + "
    f"{n_fuzzy_dup} fuzzy name+city duplicates (keeping most recent record) -> "
    f"{len(leg_clean)} clean rows")


# 7. PRODUCT CATALOG CSV: reconcile with products table
cat = pd.read_csv(CATALOG_CSV)
cat.columns = [c.strip() for c in cat.columns]
cat = cat.rename(columns={
    "SKU": "product_id",
    "item_name": "catalog_name",
    "dept": "category",
    "list_price_usd": "catalog_price",
    "supplier_cost": "catalog_cost",
    "in_stock_units": "stock_units",
})

db_ids = set(products_clean["product_id"])
csv_ids = set(cat["product_id"])
only_in_db = db_ids - csv_ids
only_in_csv = csv_ids - db_ids
in_both = db_ids & csv_ids

products_merged = products_clean.merge(
    cat[["product_id", "catalog_price", "catalog_cost", "stock_units"]],
    on="product_id", how="left"
)
products_merged["in_supplier_catalog"] = products_merged["product_id"].isin(csv_ids)

log(f"[product_catalog] DB has {len(db_ids)} products, CSV has {len(csv_ids)} SKUs; "
    f"{len(in_both)} overlap, {len(only_in_db)} DB-only (no supplier record), "
    f"{len(only_in_csv)} supplier-only SKUs not in the database at all "
    "(these represent products UrbanCart could source but has never sold -- "
    "kept in a separate table, not merged into products, since they have no "
    "order/review history to analyze)")

supplier_only = cat.loc[cat["product_id"].isin(only_in_csv)].reset_index(drop=True)


# 8. RESHAPE: category x month revenue pivot table
oi_rev = order_items_clean.merge(orders[["order_id", "order_date"]], on="order_id")
oi_rev = oi_rev.merge(products_clean[["product_id", "category"]], on="product_id")
oi_rev["year_month"] = oi_rev["order_date"].dt.to_period("M").astype(str)

category_month_pivot = pd.pivot_table(
    oi_rev.loc[~oi_rev["is_return"]],
    values="line_amount", index="category", columns="year_month",
    aggfunc="sum", fill_value=0
).round(2)
log(f"[reshape] category x month revenue pivot table: {category_month_pivot.shape}")


# 9. TIME SERIES: weekly active customers via resample
active = orders.set_index("order_date").sort_index()
weekly_active_customers = active["customer_id"].resample("W").nunique()
weekly_active_customers.name = "active_customers"
log(f"[time series] weekly active customers series: {len(weekly_active_customers)} weeks, "
    f"range {weekly_active_customers.min()}-{weekly_active_customers.max()}")


# 10. SAVE PROCESSED OUTPUTS
import os
os.makedirs(OUT_DIR, exist_ok=True)

customers_clean.to_csv(f"{OUT_DIR}/clean_customers.csv", index=False)
products_merged.to_csv(f"{OUT_DIR}/clean_products.csv", index=False)
orders.to_csv(f"{OUT_DIR}/clean_orders.csv", index=False)
order_items_clean.to_csv(f"{OUT_DIR}/clean_order_items.csv", index=False)
reviews_clean.to_csv(f"{OUT_DIR}/clean_reviews.csv", index=False)
web_sessions.to_csv(f"{OUT_DIR}/clean_web_sessions.csv", index=False)
leg_clean.to_csv(f"{OUT_DIR}/clean_legacy_customers.csv", index=False)
supplier_only.to_csv(f"{OUT_DIR}/supplier_only_skus.csv", index=False)
category_month_pivot.to_csv(f"{OUT_DIR}/category_month_pivot.csv")
weekly_active_customers.to_csv(f"{OUT_DIR}/weekly_active_customers.csv")

with open(f"{OUT_DIR}/cleaning_log.txt", "w") as f:
    f.write("\n".join(log_lines))

log(f"\n[DONE] All processed files written to {OUT_DIR}/")
