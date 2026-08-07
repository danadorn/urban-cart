

-- Query 1: Total revenue, order count, and average order value 
SELECT
    p.category,
    COUNT(DISTINCT o.order_id)                                   AS order_count,
    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS total_revenue,
    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount))
          / COUNT(DISTINCT o.order_id), 2)                        AS avg_order_value
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN orders o   ON o.order_id   = oi.order_id
WHERE oi.quantity > 0
GROUP BY p.category
ORDER BY total_revenue DESC;


-- Query 2: Top 20 customers by lifetime spend, with city and signup date
SELECT
    c.customer_id,
    c.name,
    c.city,
    c.signup_date,
    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS lifetime_spend
FROM customers c
JOIN orders o     ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
WHERE oi.quantity > 0
GROUP BY c.customer_id, c.name, c.city, c.signup_date
ORDER BY lifetime_spend DESC
LIMIT 20;


-- Query 3: Month-over-month revenue trend for the last 24 months
WITH monthly AS (
    SELECT
        strftime('%Y-%m', o.order_date) AS year_month,
        SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE oi.quantity > 0
    GROUP BY year_month
)
SELECT
    year_month,
    ROUND(revenue, 2) AS revenue,
    ROUND(LAG(revenue) OVER (ORDER BY year_month), 2)              AS prev_month_revenue,
    ROUND(revenue - LAG(revenue) OVER (ORDER BY year_month), 2)    AS mom_change,
    ROUND(SUM(revenue) OVER (ORDER BY year_month), 2)              AS running_total
FROM monthly
ORDER BY year_month DESC
LIMIT 24;


-- Query 4: Return rate per product category (share of order_items rows with negative quantity)
WITH category_items AS (
    SELECT
        p.category,
        oi.quantity
    FROM order_items oi
    JOIN products p ON p.product_id = oi.product_id
)
SELECT
    category,
    COUNT(*)                                          AS total_line_items,
    SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END)      AS return_line_items,
    ROUND(1.0 * SUM(CASE WHEN quantity < 0 THEN 1 ELSE 0 END) / COUNT(*), 4) AS return_rate
FROM category_items
GROUP BY category
ORDER BY return_rate DESC;


-- Query 5: Customers who placed orders in every one of the last 3
WITH order_quarters AS (
    SELECT DISTINCT
        customer_id,
        CASE
            WHEN strftime('%m', order_date) IN ('01','02','03') THEN strftime('%Y', order_date) || '-Q1'
            WHEN strftime('%m', order_date) IN ('04','05','06') THEN strftime('%Y', order_date) || '-Q2'
            WHEN strftime('%m', order_date) IN ('07','08','09') THEN strftime('%Y', order_date) || '-Q3'
            ELSE strftime('%Y', order_date) || '-Q4'
        END AS year_quarter
    FROM orders
    WHERE order_date >= '2024-04-01'
)
SELECT
    customer_id,
    COUNT(DISTINCT year_quarter) AS quarters_active
FROM order_quarters
WHERE year_quarter IN ('2024-Q2', '2024-Q3', '2024-Q4')
GROUP BY customer_id
HAVING COUNT(DISTINCT year_quarter) = 3
ORDER BY customer_id;


-- Query 6: 10 products with the highest average review rating among
SELECT
    p.product_id,
    p.name,
    p.category,
    COUNT(r.review_id)             AS review_count,
    ROUND(AVG(r.rating), 2)        AS avg_rating
FROM products p
JOIN reviews r ON r.product_id = p.product_id
WHERE r.rating BETWEEN 1 AND 5   -- guard against out-of-range rating values
GROUP BY p.product_id, p.name, p.category
HAVING COUNT(r.review_id) >= 15
ORDER BY avg_rating DESC, review_count DESC
LIMIT 10;


-- Query 7: Average session duration and pages viewed by device type
SELECT
    ws.device,
    ROUND(AVG(ws.duration_minutes), 2) AS avg_duration_minutes,
    ROUND(AVG(ws.pages_viewed), 2)     AS avg_pages_viewed,
    COUNT(*)                           AS session_count
FROM web_sessions ws
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = ws.customer_id
)
GROUP BY ws.device
ORDER BY avg_duration_minutes DESC;


-- Query 8: RANK() of products by revenue within each category.
WITH product_revenue AS (
    SELECT
        p.category,
        p.product_id,
        p.name,
        SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue
    FROM order_items oi
    JOIN products p ON p.product_id = oi.product_id
    WHERE oi.quantity > 0
    GROUP BY p.category, p.product_id, p.name
)
SELECT
    category,
    product_id,
    name,
    ROUND(revenue, 2) AS revenue,
    RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS revenue_rank
FROM product_revenue
ORDER BY category, revenue_rank
LIMIT 60;  -- top ranks across all 6 categories, trim/filter as needed in report


-- Query 9: Payment-method mix (share of orders) split by country.
WITH country_totals AS (
    SELECT c.country, COUNT(*) AS total_orders
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    GROUP BY c.country
)
SELECT
    c.country,
    o.payment_method,
    COUNT(*)                                                    AS order_count,
    ROUND(1.0 * COUNT(*) / ct.total_orders, 4)                  AS share_of_country_orders
FROM orders o
JOIN customers c       ON c.customer_id = o.customer_id
JOIN country_totals ct ON ct.country    = c.country
GROUP BY c.country, o.payment_method
ORDER BY c.country, share_of_country_orders DESC;


-- Query 10 (custom): Which cities have the highest revenue-per-customer
WITH city_customers AS (
    SELECT city, COUNT(*) AS n_customers
    FROM customers
    WHERE city IS NOT NULL AND TRIM(city) != ''
    GROUP BY city
    HAVING COUNT(*) >= 20
),
city_revenue AS (
    SELECT
        c.city,
        SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue
    FROM customers c
    JOIN orders o      ON o.customer_id = c.customer_id
    JOIN order_items oi ON oi.order_id  = o.order_id
    WHERE oi.quantity > 0
    GROUP BY c.city
)
SELECT
    cc.city,
    cc.n_customers,
    ROUND(cr.revenue, 2)                          AS total_revenue,
    ROUND(cr.revenue / cc.n_customers, 2)         AS revenue_per_customer
FROM city_customers cc
JOIN city_revenue cr ON cr.city = cc.city
ORDER BY revenue_per_customer DESC;
