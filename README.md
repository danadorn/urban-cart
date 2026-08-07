# UrbanCart Analytics Term Project

A complete analytics pipeline for UrbanCart, a fictional mid-size online retailer — moving
from raw, imperfect data (a SQLite database plus two messy external CSVs) to a decision-ready
30-page report, covering SQL extraction, pandas cleaning/integration, NumPy statistical
routines, and business visualizations.

## Repository Structure

```
queries.sql                  # Phase 1: all 10 SQL queries, run against ecommerce.db
phase2_cleaning.py           # Phase 2: pandas cleaning & integration of DB + 2 CSVs
phase3_numpy.py              # Phase 3: RFM, cosine similarity, regression, Monte Carlo (raw NumPy)
phase4_visuals.py            # Phase 4: 8 charts answering the business questions
analysis.ipynb               # All of the above combined into one notebook
data/processed/              # Cleaned, saved output tables (csv)
figures/                     # Generated charts (png)
report.docx / report.pdf     # Full 30-page written report
executive_summary.pdf/docx   # 1-page summary for non-technical leadership
README_DATA.md               # Original data dictionary (provided with raw files)
```

## Reproducing the Pipeline

```bash
pip install pandas numpy matplotlib seaborn --break-system-packages
python phase2_cleaning.py   # reads raw files, writes data/processed/*.csv
python phase3_numpy.py      # reads data/processed/, writes RFM/similarity/regression/Monte Carlo outputs
python phase4_visuals.py    # reads data/processed/, writes figures/*.png
```

Each stage only reads the previous stage's saved output, so any stage can be re-run in
isolation once its upstream dependency is up to date.

`phase2_cleaning.py` also loads 4 of the Phase 1 SQL queries (category revenue, top-20
customers, return rate by category, payment mix by country) directly into pandas via
`pandas.read_sql`, saving them to `data/processed/sql_*.csv` and using the return-rate
query as a live cross-check against the pandas-computed return flag — these are executed
queries wired into the Python pipeline, not pasted output tables.

## Individual Contribution Statement

This project was completed independently. All four phases — SQL extraction (Phase 1),
pandas-based data cleaning and integration of the legacy CRM export and supplier product
catalog with the operational database (Phase 2), the four required NumPy statistical
routines including RFM segmentation, cosine-similarity product recommendations, normal-equation
regression, and Monte Carlo stockout simulation (Phase 3), and the resulting business
visualizations and written report (Phase 4) — were carried out by the sole author of this
repository. Data-cleaning decisions (missing-value policy, outlier thresholds, deduplication
rules) were made and documented directly in the pipeline scripts and in Section 4 of the
written report, with every choice logged automatically to `data/processed/cleaning_log.txt`
on each pipeline run for auditability. No portion of the codebase or report was produced by
a team member other than the author.

## Notes

- Revenue figures throughout use *net* revenue (return line-items included) unless explicitly
  labeled "gross" — see Section 4.8 and Appendix E of the report for why this matters.
- The dataset is treated as authentic: known data-quality issues (duplicate rows, out-of-range
  ratings, inconsistent date formats, price outliers) are documented and handled explicitly
  rather than assumed away.
